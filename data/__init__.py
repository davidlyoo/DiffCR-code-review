from functools import partial

import numpy as np
from torch import Generator, randperm
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

import core.util as Util
from core.praser import init_obj

# =============================================================================
# 데이터셋 및 DataLoader 생성 관련 함수들
# - define_dataloader: 학습/테스트 및 검증용 DataLoader를 생성
# - define_dataset: 옵션(opt)에 따라 데이터셋 객체들을 생성
# - subset_split: 데이터셋을 주어진 길이로 비중복 부분집합으로 분할 (pytorch의 random_split과 유사)
# =============================================================================


def define_dataloader(logger, opt):
    """
    학습/테스트용 DataLoader와 검증용 DataLoader를 생성
    - 검증 DataLoader는 phase가 test거나 GPU 0이 아니면 None을 반환함

    주요 단계:
      1. 옵션(opt)에서 dataloader 관련 인자와 worker 초기 시드를 설정
      2. define_dataset()을 호출해 주 데이터셋과 검증 데이터셋 생성
      3. 분산 학습(distributed)이 활성화되어 있으면 DistributedSampler로 샘플러 생성
         (sampler 사용시 shuffle 옵션은 False로 설정)
      4. main DataLoader와, 만약 GPU 0이고 검증 데이터셋이 존재하면 검증용 DataLoader 생성
    """
    dataloader_args = opt['datasets'][opt['phase']]['dataloader']['args']
    worker_init_fn = partial(Util.set_seed, gl_seed=opt['seed'])

    phase_dataset, val_dataset = define_dataset(logger, opt)

    # 분산 학습 환경에서 main dataset에 대한 샘플러 생성 (shuffle=False)
    data_sampler = None
    if opt['distributed']:
        data_sampler = DistributedSampler(phase_dataset, shuffle=dataloader_args.get('shuffle', False),
                                            num_replicas=opt['world_size'], rank=opt['global_rank'])
        dataloader_args.update({'shuffle': False})

    # 학습/테스트용 DataLoader 생성
    dataloader = DataLoader(phase_dataset, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)

    # 검증 DataLoader: GPU 0에서만 사용 (opt['global_rank'] == 0) // 검증 데이터셋이 존재해야 함
    if opt['global_rank'] == 0 and val_dataset is not None:
        dataloader_args.update(opt['datasets'][opt['phase']]['dataloader'].get('val_args', {}))
        val_dataloader = DataLoader(val_dataset, worker_init_fn=worker_init_fn, **dataloader_args)
    else:
        val_dataloader = None
    return dataloader, val_dataloader


def define_dataset(logger, opt):
    """
    옵션(opt)에 따라 학습/테스트용과 검증용 데이터셋 객체 생성
    - dataset_opt와 val_dataset_opt 정보를 바탕으로 core.praser 모듈의 init_obj()를 호출하여 데이터셋 인스턴스 생성
    - 디버그 모드라면 데이터셋 길이를 조정
    - 생성된 데이터셋의 샘플 수를 logger.info()로 출력
    """
    dataset_opt = opt['datasets'][opt['phase']]['which_dataset']
    phase_dataset = init_obj(dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset')

    val_dataset_opt = opt['datasets']['val']['which_dataset']
    val_dataset = init_obj(val_dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset')

    valid_len = 0
    data_len = len(phase_dataset)
    if 'debug' in opt['name']:
        debug_split = opt['debug'].get('debug_split', 1.0)
        if isinstance(debug_split, int):
            data_len = debug_split
        else:
            data_len *= debug_split

    logger.info('Dataset for {} have {} samples.'.format(opt['phase'], data_len))
    if opt['phase'] == 'train':
        logger.info('Dataset for {} have {} samples.'.format('val', valid_len))
    return phase_dataset, val_dataset


def subset_split(dataset, lengths, generator):
    """
    주어진 길이(lengths)로 데이터셋을 비중복(subset)으로 분할하는 함수
    - 내부적으로 torch.randperm를 이용해 인덱스 순서를 무작위로 섞은 후, 지정된 길이에 따라 Subset 객체들을 생성
    - pytorch의 random_split과 유사하게 동작
    """
    indices = randperm(sum(lengths), generator=generator).tolist()
    Subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths):
        if length == 0:
            Subsets.append(None)
        else:
            Subsets.append(Subset(dataset, indices[offset - length : offset]))
    return Subsets