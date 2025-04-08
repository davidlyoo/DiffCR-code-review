# =============================================================================
# core/logger.py
# - 파일의 역할: 로그 기록, 시각화, 훈련 수치 지표 추적 기능을 제공함
# - InfoLogger: Python의 logging 모듈을 이용해 로그를 기록하며, 분산 학습 환경에서 GPU 0에서만 동작하도록 제어함
# - VisualWriter: Tensorboard를 활용해 시각적 정보를 기록하고 결과 이미지를 저장하는 기능 제공
# - LogTracker: 훈련 중 주요 수치 지표(예: 손실, 정확도 등)를 pandas DataFrame을 이용해 관리 및 추적
# =============================================================================

import os
from PIL import Image
import importlib
from datetime import datetime
import logging
import pandas as pd

import core.util as Util


# =============================================================================
# InfoLogger 클래스
# - 분산 학습 환경에서 global rank 0인 경우에만 로그를 기록하도록 하는 래퍼 클래스
# =============================================================================
class InfoLogger():
    """
    로그 기록에 logging 모듈을 사용하며, 분산 학습 시 GPU 0에서만 로그를 출력함
    """
    def __init__(self, opt):
        self.opt = opt
        self.rank = opt['global_rank']
        self.phase = opt['phase']

        # 실험 결과 로그 파일을 생성 및 설정 (opt의 경로, phase 정보 사용)
        self.setup_logger(None, opt['path']['experiments_root'], opt['phase'], level=logging.INFO, screen=False)
        self.logger = logging.getLogger(opt['phase'])
        # 기록하고자 하는 로그 함수 목록
        self.infologger_ftns = {'info', 'warning', 'debug'}

    def __getattr__(self, name):
        # GPU 0가 아니면 로그 출력을 무시하기 위해 빈 함수를 반환
        if self.rank != 0:
            def wrapper(info, *args, **kwargs):
                pass
            return wrapper
        # GPU 0인 경우, 요청한 로그 함수(name)에 대해 logger의 해당 함수를 호출하는 래퍼를 반환
        if name in self.infologger_ftns:
            print_info = getattr(self.logger, name, None)
            def wrapper(info, *args, **kwargs):
                print_info(info, *args, **kwargs)
            return wrapper

    @staticmethod
    def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
        """
        로그 설정 함수
         - 지정한 경로(root) 아래에 phase 이름의 log 파일을 생성하고 포매터를 적용함
         - 옵션으로 로그를 콘솔에도 출력(screen=True)할 수 있음
        """
        l = logging.getLogger(logger_name)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        log_file = os.path.join(root, '{}.log'.format(phase))
        fh = logging.FileHandler(log_file, mode='a+')
        fh.setFormatter(formatter)
        l.setLevel(level)
        l.addHandler(fh)
        if screen:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            l.addHandler(sh)


# =============================================================================
# VisualWriter 클래스
# - Tensorboard를 사용하여 시각적 로그 (scalar, image, audio 등)를 기록하고, 
#   결과 이미지를 지정된 경로에 저장하는 기능을 제공함
# =============================================================================
class VisualWriter():
    """ 
    Tensorboard를 사용하여 시각적 로그를 기록하며, 결과 저장 기능과 통합되어 있음
    """
    def __init__(self, opt, logger):
        log_dir = opt['path']['tb_logger']
        self.result_dir = opt['path']['results']
        enabled = opt['train']['tensorboard']
        self.rank = opt['global_rank']

        self.writer = None
        self.selected_module = ""

        if enabled and self.rank == 0:
            log_dir = str(log_dir)
            # Tensorboard writer를 임포트하여 SummaryWriter 객체 생성 시도
            succeeded = False
            for module in ["tensorboardX", "torch.utils.tensorboard"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = ("Warning: visualization (Tensorboard) is configured to use, but currently not installed on "
                           "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to "
                           "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file.")
                logger.warning(message)

        self.epoch = 0
        self.iter = 0
        self.phase = ''

        # Tensorboard에 기록 가능한 함수 목록
        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        # 특정 함수는 tag 처리 방식을 예외로 함
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.custom_ftns = {'close'}
        self.timer = datetime.now()

    def set_iter(self, epoch, iter, phase='train'):
        """
        현재 진행 중인 epoch, iter, phase 정보를 설정함
        """
        self.phase = phase
        self.epoch = epoch
        self.iter = iter

    def save_images(self, results):
        """
        results(보통 OrderedDict 형태)를 받아, 저장 경로에 각 이미지 파일(PIL.Image)을 저장함
        - results에는 'name'과 'result' 키가 있어야 함
        """
        result_path = os.path.join(self.result_dir, self.phase)
        os.makedirs(result_path, exist_ok=True)
        result_path = os.path.join(result_path, str(self.epoch))
        os.makedirs(result_path, exist_ok=True)

        # 결과 dict로부터 이름과 결과 이미지 목록 추출 후 저장
        try:
            names = results['name']
            outputs = Util.postprocess(results['result'])
            for i in range(len(names)): 
                if os.path.exists(os.path.join(result_path, names[i])):
                    pass
                else:
                    Image.fromarray(outputs[i]).save(os.path.join(result_path, names[i]))
        except:
            raise NotImplementedError('You must specify the context of name and result in save_current_results functions of model.')

    def close(self):
        """
        Tensorboard SummaryWriter를 닫는 함수
        """
        self.writer.close()
        print('Close the Tensorboard SummaryWriter.')
        
    def __getattr__(self, name):
        """
        만약 Tensorboard 기록 함수(tb_writer_ftns)에 접근하면,
        phase 및 iter 정보를 추가하여 해당 함수를 호출하는 래퍼를 반환함
        만약 해당 함수가 없으면 기본 속성 접근을 시도함
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)
            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # 기본적으로 tag에 phase 정보를 추가하여, 예: 'train/your_tag' 형식으로 변환
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(self.phase, tag)
                    add_data(tag, data, self.iter, *args, **kwargs)
            return wrapper
        else:
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


# =============================================================================
# LogTracker 클래스
# - 훈련 중 주요 수치 지표(예: 손실, 정확도 등)를 pandas DataFrame을 통해 누적, 평균, 개수를 계산하여 추적함
# =============================================================================
class LogTracker:
    """
    훈련 수치 지표들을 기록하는 로그 트래커
    """
    def __init__(self, *keys, phase='train'):
        self.phase = phase
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        """
        모든 수치 지표를 0으로 초기화함
        """
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        """
        특정 key(수치 지표)에 대해 총합, 카운트 및 평균을 업데이트함
        """
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        """
        특정 key의 평균 값을 반환함
        """
        return self._data.average[key]

    def result(self):
        """
        phase 태그를 포함한 각 지표의 평균 값을 딕셔너리 형태로 반환함
        """
        return {'{}/{}'.format(self.phase, k): v for k, v in dict(self._data.average).items()}
