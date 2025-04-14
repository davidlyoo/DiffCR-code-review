# =============================================================================
# data/dataset.py
# - 이 파일은 이미지 파일 목록 생성 및 데이터셋(Dataset) 클래스를 정의함
# - 이미지 파일을 읽어들이는 유틸 함수
# - 각 모드(train/val/test)에 따라 데이터를 로드하는 Dataset 클래스들을 포함
# =============================================================================

import os
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import tifffile as tiff
from PIL import Image

# from .util.mask import (bbox2mask, brush_stroke_mask,
#                         get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    # 만약 주어진 경로가 파일이면 텍스트 파일로부터 이미지 경로 목록을 로드
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        # 디렉토리 순회하며 이미지 파일 찾기
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')

# -----------------------------------------------------------------------------
# Sen2_MTC 데이터셋 클래스 (다양한 버전)
#
# Sen2_MTC_New_Multi:
# - Sen2_MTC 데이터셋의 Multi-Modal 버전
# - 'cloud' 이미지 3개와 'cloudless' 이미지를 읽어, gt_image와 cond_image를 구성함
# - 학습 시 augmentation (회전, flip) 파라미터를 랜덤하게 부여
# 
# Sen2_MTC_New1:
# - Sen2_MTC 데이터셋의 또 다른 버전
# - 학습 모드의 경우 조건 이미지로 3개 중 하나를 랜덤 선택, 검증 등에서는 첫 번째를 사용
#
# Sen2_MTC_New2:
# - Sen2_MTC 데이터셋의 또 다른 버전
# - 'train' 모드에서는 조건 이미지로 세 장을 torch.cat하여 사용, test/val 모드에서는 두 장을 사용
# -----------------------------------------------------------------------------

class Sen2_MTC_New_Multi(data.Dataset):
    def __init__(self, data_root, mode='train'):
        self.data_root = data_root
        self.mode = mode
        self.filepair = []
        self.image_name = []

        if mode == 'train':
            self.tile_list = np.loadtxt(os.path.join(self.data_root, 'train.txt'), dtype=str)
        elif mode == 'val':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(os.path.join(self.data_root, 'val.txt'), dtype=str)
        elif mode == 'test':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(os.path.join(self.data_root, 'test.txt'), dtype=str)

        # 각 tile 별로 cloudless 폴더 내 파일명을 읽고, cloud 폴더의 3개 밴드와 cloudless 이미지를 파일경로로 구성
        for tile in self.tile_list:
            image_name_list = [image_name.split('.')[0] for image_name in os.listdir(
                os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloudless'))]
            for image_name in image_name_list:
                image_cloud_path0 = os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_0.tif')
                image_cloud_path1 = os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_1.tif')
                image_cloud_path2 = os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_2.tif')
                image_cloudless_path = os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloudless', image_name + '.tif')
                self.filepair.append([image_cloud_path0, image_cloud_path1, image_cloud_path2, image_cloudless_path])
                self.image_name.append(image_name)
        
        self.augment_rotation_param = np.random.randint(0, 4, len(self.filepair))
        self.augment_flip_param = np.random.randint(0, 3, len(self.filepair))
        self.index = 0

    def __getitem__(self, index):
        # 파일 경로들을 불러와 이미지 읽기 수행
        cloud_image_path0, cloud_image_path1, cloud_image_path2 = \
            self.filepair[index][0], self.filepair[index][1], self.filepair[index][2]
        cloudless_image_path = self.filepair[index][3]
        image_cloud0 = self.image_read(cloud_image_path0)
        image_cloud1 = self.image_read(cloud_image_path1)
        image_cloud2 = self.image_read(cloud_image_path2)
        image_cloudless = self.image_read(cloudless_image_path)

        ret = {}
        ret['gt_image'] = image_cloudless[:3, :, :]
        ret['cond_image'] = torch.cat([image_cloud0[:3, :, :], image_cloud1[:3, :, :], image_cloud2[:3, :, :]])
        ret['path'] = self.image_name[index] + ".png"
        return ret

    def __len__(self):
        return len(self.filepair)

    def image_read(self, image_path):
        img = tiff.imread(image_path)
        img = (img / 1.0).transpose((2, 0, 1))
        if self.mode == 'train':
            if not self.augment_flip_param[self.index // 4] == 0:
                img = np.flip(img, self.augment_flip_param[self.index // 4])
            if not self.augment_rotation_param[self.index // 4] == 0:
                img = np.rot90(img, self.augment_rotation_param[self.index // 4], (1, 2))
            self.index += 1
        if self.index // 4 >= len(self.filepair):
            self.index = 0
        image = torch.from_numpy(img.copy()).float()
        image = image / 10000.0
        mean = torch.as_tensor([0.5, 0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.5, 0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)
        return image


class Sen2_MTC_New1(data.Dataset):
    def __init__(self, data_root, mode='train'):
        self.data_root = data_root
        self.mode = mode
        self.filepair = []
        self.image_name = []

        if mode == 'train':
            self.tile_list = np.loadtxt(os.path.join(self.data_root, 'train.txt'), dtype=str)
        elif mode == 'val':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(os.path.join(self.data_root, 'val.txt'), dtype=str)
        elif mode == 'test':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(os.path.join(self.data_root, 'test.txt'), dtype=str)

        for tile in self.tile_list:
            image_name_list = [image_name.split('.')[0] for image_name in os.listdir(
                os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloudless'))]
            for image_name in image_name_list:
                image_cloud_path0 = os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_0.tif')
                image_cloud_path1 = os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_1.tif')
                image_cloud_path2 = os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_2.tif')
                image_cloudless_path = os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloudless', image_name + '.tif')
                self.filepair.append([image_cloud_path0, image_cloud_path1, image_cloud_path2, image_cloudless_path])
                self.image_name.append(image_name)

        self.augment_rotation_param = np.random.randint(0, 4, len(self.filepair))
        self.augment_flip_param = np.random.randint(0, 3, len(self.filepair))
        self.index = 0

    def __getitem__(self, index):
        cloud_image_path0, cloud_image_path1, cloud_image_path2 = \
            self.filepair[index][0], self.filepair[index][1], self.filepair[index][2]
        cloudless_image_path = self.filepair[index][3]
        image_cloud0 = self.image_read(cloud_image_path0)
        image_cloud1 = self.image_read(cloud_image_path1)
        image_cloud2 = self.image_read(cloud_image_path2)
        image_cloudless = self.image_read(cloudless_image_path)
        
        ret = {}
        ret['gt_image'] = image_cloudless[:3, :, :]
        if self.mode == 'train':
            ret['cond_image'] = random.choice([image_cloud0, image_cloud1, image_cloud2])[:3, :, :]
        else:
            ret['cond_image'] = image_cloud0[:3, :, :]
        ret['path'] = self.image_name[index] + ".png"
        return ret

    def __len__(self):
        return len(self.filepair)

    def image_read(self, image_path):
        img = tiff.imread(image_path)
        img = (img / 1.0).transpose((2, 0, 1))
        if self.mode == 'train':
            if not self.augment_flip_param[self.index // 4] == 0:
                img = np.flip(img, self.augment_flip_param[self.index // 4])
            if not self.augment_rotation_param[self.index // 4] == 0:
                img = np.rot90(img, self.augment_rotation_param[self.index // 4], (1, 2))
            self.index += 1
        if self.index // 4 >= len(self.filepair):
            self.index = 0
        image = torch.from_numpy(img.copy()).float()
        image = image / 10000.0
        mean = torch.as_tensor([0.5, 0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.5, 0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)
        return image


class Sen2_MTC_New2(data.Dataset):
    def __init__(self, data_root, mode='train'):
        self.data_root = data_root
        self.mode = mode
        self.filepair = []
        self.image_name = []
        
        if mode == 'train':
            self.tile_list = np.loadtxt(os.path.join(self.data_root, 'train.txt'), dtype=str)
        elif mode == 'val':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(os.path.join(self.data_root, 'val.txt'), dtype=str)
        elif mode == 'test':
            self.data_augmentation = None
            self.tile_list = np.loadtxt(os.path.join(self.data_root, 'test.txt'), dtype=str)

        for tile in self.tile_list:
            image_name_list = [image_name.split('.')[0] for image_name in os.listdir(
                os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloudless'))]
            for image_name in image_name_list:
                image_cloud_path0 = os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_0.tif')
                image_cloud_path1 = os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_1.tif')
                image_cloud_path2 = os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloud', image_name + '_2.tif')
                image_cloudless_path = os.path.join(self.data_root, 'Sen2_MTC', tile, 'cloudless', image_name + '.tif')
                self.filepair.append([image_cloud_path0, image_cloud_path1, image_cloud_path2, image_cloudless_path])
                self.image_name.append(image_name)

        self.augment_rotation_param = np.random.randint(0, 4, len(self.filepair))
        self.augment_flip_param = np.random.randint(0, 3, len(self.filepair))
        self.index = 0

    def __getitem__(self, index):
        cloud_image_path0, cloud_image_path1, cloud_image_path2 = \
            self.filepair[index][0], self.filepair[index][1], self.filepair[index][2]
        cloudless_image_path = self.filepair[index][3]
        image_cloud0 = self.image_read(cloud_image_path0)
        image_cloud1 = self.image_read(cloud_image_path1)
        image_cloud2 = self.image_read(cloud_image_path2)
        image_cloudless = self.image_read(cloudless_image_path)

        ret = {}
        ret['gt_image'] = image_cloudless[:3, :, :]
        # 훈련 시에는 세 이미지를 연결한 후 랜덤 샘플링, 그렇지 않으면 앞 두 이미지 사용
        if self.mode=="train":
            ret['cond_image'] = torch.cat(random.sample(
                (image_cloud0[:3, :, :], image_cloud1[:3, :, :], image_cloud2[:3, :, :]), 2))
        else:
            ret['cond_image'] = torch.cat([image_cloud0[:3, :, :], image_cloud1[:3, :, :]])
        ret['path'] = self.image_name[index] + ".png"
        return ret

    def __len__(self):
        return len(self.filepair)

    def image_read(self, image_path):
        img = tiff.imread(image_path)
        img = (img / 1.0).transpose((2, 0, 1))
        if self.mode == 'train':
            if not self.augment_flip_param[self.index // 4] == 0:
                img = np.flip(img, self.augment_flip_param[self.index // 4])
            if not self.augment_rotation_param[self.index // 4] == 0:
                img = np.rot90(img, self.augment_rotation_param[self.index // 4], (1, 2))
            self.index += 1
        if self.index // 4 >= len(self.filepair):
            self.index = 0
        image = torch.from_numpy(img.copy()).float()
        image = image / 10000.0
        mean = torch.as_tensor([0.5, 0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.5, 0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)
        return image

# -----------------------------------------------------------------------------
# ColorizationDataset 클래스
# - 색상 복원(colorization) 작업용 데이터셋
# - 'color' 이미지와 'gray' 이미지 쌍을 읽어, 조건 이미지(cond_image)와 gt_image를 구성함
# -----------------------------------------------------------------------------
class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'
        img = self.tfs(self.loader(f'{self.data_root}/color/{file_name}'))
        cond_image = self.tfs(self.loader(f'{self.data_root}/gray/{file_name}'))
        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)

# -----------------------------------------------------------------------------
# InpaintDataset 클래스
# - 이미지 파일 목록을 생성하고, 각 이미지에 대해 마스크를 생성하여
#   조건 이미지(cond_image)를 생성 (마스크 영역은 random noise 사용)
# - gt_image, cond_image, mask_image, mask, path 정보를 반환
# -----------------------------------------------------------------------------
# class InpaintDataset(data.Dataset):
#     def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
#         imgs = make_dataset(data_root)
#         if data_len > 0:
#             self.imgs = imgs[:int(data_len)]
#         else:
#             self.imgs = imgs
#         self.tfs = transforms.Compose([
#             transforms.Resize((image_size[0], image_size[1])),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
#         self.loader = loader
#         self.mask_config = mask_config
#         self.mask_mode = self.mask_config['mask_mode']
#         self.image_size = image_size

#     def __getitem__(self, index):
#         ret = {}
#         path = self.imgs[index]
#         img = self.tfs(self.loader(path))
#         mask = self.get_mask()
#         cond_image = img * (1. - mask) + mask * torch.randn_like(img)
#         mask_img = img * (1. - mask) + mask

#         ret['gt_image'] = img
#         ret['cond_image'] = cond_image
#         ret['mask_image'] = mask_img
#         ret['mask'] = mask
#         ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
#         return ret

#     def __len__(self):
#         return len(self.imgs)

#     def get_mask(self):
#         if self.mask_mode == 'bbox':
#             mask = bbox2mask(self.image_size, random_bbox())
#         elif self.mask_mode == 'center':
#             h, w = self.image_size
#             mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
#         elif self.mask_mode == 'irregular':
#             mask = get_irregular_mask(self.image_size)
#         elif self.mask_mode == 'free_form':
#             mask = brush_stroke_mask(self.image_size)
#         elif self.mask_mode == 'hybrid':
#             regular_mask = bbox2mask(self.image_size, random_bbox())
#             irregular_mask = brush_stroke_mask(self.image_size)
#             mask = regular_mask | irregular_mask
#         elif self.mask_mode == 'file':
#             pass
#         else:
#             raise NotImplementedError(f'Mask mode {self.mask_mode} has not been implemented.')
#         return torch.from_numpy(mask).permute(2, 0, 1)

# -----------------------------------------------------------------------------
# UncroppingDataset 클래스
# - InpaintDataset와 유사하며, 마스크 생성 방식에 따라 이미지 복원을 위한 데이터셋 구성
# -----------------------------------------------------------------------------
# class UncroppingDataset(data.Dataset):
#     def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
#         imgs = make_dataset(data_root)
#         if data_len > 0:
#             self.imgs = imgs[:int(data_len)]
#         else:
#             self.imgs = imgs
#         self.tfs = transforms.Compose([
#             transforms.Resize((image_size[0], image_size[1])),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
#         self.loader = loader
#         self.mask_config = mask_config
#         self.mask_mode = self.mask_config['mask_mode']
#         self.image_size = image_size

#     def __getitem__(self, index):
#         ret = {}
#         path = self.imgs[index]
#         img = self.tfs(self.loader(path))
#         mask = self.get_mask()
#         cond_image = img * (1. - mask) + mask * torch.randn_like(img)
#         mask_img = img * (1. - mask) + mask

#         ret['gt_image'] = img
#         ret['cond_image'] = cond_image
#         ret['mask_image'] = mask_img
#         ret['mask'] = mask
#         ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
#         return ret

#     def __len__(self):
#         return len(self.imgs)

#     def get_mask(self):
#         if self.mask_mode == 'manual':
#             mask = bbox2mask(self.image_size, self.mask_config['shape'])
#         elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
#             mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
#         elif self.mask_mode == 'hybrid':
#             if np.random.randint(0, 2) < 1:
#                 mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
#             else:
#                 mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
#         elif self.mask_mode == 'file':
#             pass
#         else:
#             raise NotImplementedError(f'Mask mode {self.mask_mode} has not been implemented.')
#         return torch.from_numpy(mask).permute(2, 0, 1)


if __name__=='__main__':
    import numpy as np
    # 테스트용 get_rgb_tensor 함수 예시 (Tensor → 이미지 변환)
    def get_rgb_tensor(image):
        image = image * 0.5 + 0.5
        rgb = image[:3, :, :]
        rgb = rgb - torch.min(rgb)
        if torch.max(rgb) == 0:
            rgb = 255 * torch.ones_like(rgb)
        else:
            rgb = 255 * (rgb / torch.max(rgb))
        return rgb.type(torch.uint8).permute(1, 2, 0).contiguous()

    def get_rgb(image):
        image = image.mul(0.5).add_(0.5)
        image = image.squeeze()
        image = image.mul(10000).add_(0.5).clamp_(0, 10000)
        image = image.permute(1, 2, 0).cpu().detach().numpy()
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
        rgb = rgb.astype(np.uint8)
        return rgb

    # 예시: Sen2_MTC_New 데이터셋을 사용해 이미지 저장
    # (여기서는 Sen2_MTC_New 클래스 호출 시 데이터 경로 및 모드 등은 실제 사용 환경에 맞게 조정)
    for ret in Sen2_MTC_New2("datasets", "val"):
        # 예시로 cond_image를 저장
        Image.fromarray(((ret['cond_image']*0.5+0.5)*255).permute(1, 2, 0).numpy().astype(np.uint8)).save("cond.png")
        break