import math
import random
import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid

def get_rgb(image):  # CHW 형식 입력
    image = image.mul(0.5).add_(0.5)
    image = image.mul(10000).add_(0.5).clamp_(0, 10000)
    image = image.permute(1, 2, 0).cpu().detach().numpy()  # HWC 순서 변환
    image = image.astype(np.uint16)
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    r = np.clip(r, 0, 2000)
    g = np.clip(g, 0, 2000)
    b = np.clip(b, 0, 2000)
    rgb = np.dstack((r, g, b))
    rgb = rgb - np.nanmin(rgb)
    if np.nanmax(rgb) == 0:
        rgb = 255 * np.ones_like(rgb)
    else:
        rgb = 255 * (rgb / np.nanmax(rgb))
    rgb[np.isnan(rgb)] = np.nanmean(rgb)
    rgb = rgb.astype(np.uint8)
    return rgb


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    텐서를 이미지 numpy 배열로 변환 (입력: 4D, 3D, 2D 텐서, 출력: HWC 또는 2D)
    '''
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid([torch.from_numpy(get_rgb(tensor[i])).permute(2, 0, 1)
                             for i in range(n_img)], nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 3:
        img_np = get_rgb(tensor)
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor But received dimension: {:d}'.format(n_dim))
    return img_np

# ----------------------------------------------------------------------------
# 아래 주석 처리된 tensor2img 함수들은
# 원저자 또는 이전 버전에서 사용했던 대안 구현 예시
# 서로 다른 방법(예: clamp, get_rgb_tensor 등)으로 텐서를 이미지로 변환
# 필요 시 참고하여 사용할 수 있음
# ----------------------------------------------------------------------------

# Alternative version 1
# def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
#     '''
#     Converts a torch Tensor into an image Numpy array
#     Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
#     Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
#     '''
#     tensor = tensor.clamp_(*min_max)  # clamp
#     n_dim = tensor.dim()
#     if n_dim == 4:
#         n_img = len(tensor)
#         img_np = make_grid(tensor[:, :3, :, :], nrow=int(math.sqrt(n_img)), normalize=False).numpy()
#         img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
#     elif n_dim == 3:
#         img_np = tensor[:3, :, :].numpy()
#         img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
#     elif n_dim == 2:
#         img_np = tensor.numpy()
#     else:
#         raise TypeError('Only support 4D, 3D and 2D tensor But received dimension: {:d}'.format(n_dim))
#     if out_type == np.uint8:
#         img_np = ((img_np+1) * 127.5).round()
#         # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
#     return img_np.astype(out_type).squeeze()

# Alternative version 2
# def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
#     '''
#     Converts a torch Tensor into an image Numpy array
#     Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
#     Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
#     '''
#     tensor = tensor.clamp_(*min_max)  # clamp
#     def get_rgb_tensor(rgb):
#         rgb = rgb*0.5+0.5
#         rgb = rgb - torch.min(rgb)
#         # treat saturated images, scale values
#         if torch.max(rgb) == 0:
#             rgb = 255 * torch.ones_like(rgb)
#         else:
#             rgb = 255 * (rgb / torch.max(rgb))
#         return rgb.type(torch.uint8)
#     tensor = get_rgb_tensor(tensor)
#     n_dim = tensor.dim()
#     if n_dim == 4:
#         n_img = len(tensor)
#         img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
#         img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
#     elif n_dim == 3:
#         img_np = tensor.numpy()
#         img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
#     elif n_dim == 2:
#         img_np = tensor.numpy()
#     else:
#         raise TypeError('Only support 4D, 3D and 2D tensor But received dimension: {:d}'.format(n_dim))
#     return img_np.astype(out_type).squeeze()


def postprocess(images):
    return [tensor2img(image) for image in images]

# ------------------------------------------------------------
# 랜덤 시드 및 디바이스 설정 관련 함수
# - set_seed: 다양한 라이브러리의 시드를 고정해 재현성 확보
# - set_gpu / set_device: 모델 및 인자를 GPU 또는 분산 환경에 맞게 설정
# ------------------------------------------------------------

def set_seed(seed, gl_seed=0):
    if seed >= 0 and gl_seed >= 0:
        seed += gl_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    if seed >= 0 and gl_seed >= 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def set_gpu(args, distributed=False, rank=0):
    if args is None:
        return None
    if distributed and isinstance(args, torch.nn.Module):
        return DDP(args.cuda(), device_ids=[rank], output_device=rank,
                   broadcast_buffers=True, find_unused_parameters=False)
    else:
        return args.cuda()


def set_device(args, distributed=False, rank=0):
    if torch.cuda.is_available():
        if isinstance(args, list):
            return [set_gpu(item, distributed, rank) for item in args]
        elif isinstance(args, dict):
            return {key: set_gpu(args[key], distributed, rank) for key in args}
        else:
            args = set_gpu(args, distributed, rank)
    return args