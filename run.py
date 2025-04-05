import argparse
import os
import warnings

import torch
import torch.multiprocessing as mp

import core.praser as Praser
import core.util as Util
from core.logger import VisualWriter, InfoLogger
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric


# 각 GPU에서 실행될 train/test 프로세스 정의
def main_worker(gpu, ngpus_per_node, opt):
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    # 멀티 GPU 훈련 지원
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(
            backend = 'nccl',
            init_method = opt['init_method'],
            world_size = opt['world_size'],
            rank = opt['global_rank'],
            group_name = 'mtorch'
        )

    # 난수 시드 고정 및 cuDNN 최적화
    torch.backends.cudnn.enabled = True
    Util.set_seed(opt['seed'])

    # 로그 및 시각화 설정
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)
    phase_logger.info('Creat the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    # 데이터 로더, 네트워크 설정
    phase_loader, val_loader = define_dataloader(phase_logger, opt)
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    # 평가 지표, loss 설정
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which losses']]

    model = create_model(
        opt = opt,
        networks = networks,
        phase_loader = phase_loader,
        val_loader = val_loader,
        losses = losses,
        metrics = metrics,
        logger = phase_logger,
        writer = phase_writer
    )

    phase_logger.info('Begin model {}.'.format(opt['phase']))
    try:
        if opt['phase'] == 'train':
            model.train()
        else:
            model.test()
    finally:
        phase_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/colorization_mirflickr25k.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)

    # 인자 파싱 및 설정 로드
    args = parser.parse_args()
    opt = Praser.parse(args)

    # GPU 설정
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    # 멀티 GPU 사용 - DistributedDataParallel(DDP) and multiprocessing
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids'])
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:' + args.port
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1
        main_worker(0, 1, opt)