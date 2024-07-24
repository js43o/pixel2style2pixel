import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
from random import random
from torchvision.transforms import transforms
import torchvision.transforms.functional as F
import math


class ImagesDataset(Dataset):

    def __init__(
        self,
        source_root,
        target_root,
        opts,
        target_transform=None,
        source_transform=None,
    ):
        self.source_paths = sorted(data_utils.make_dataset(source_root))
        self.target_paths = sorted(data_utils.make_dataset(target_root))
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.opts = opts

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = Image.open(from_path)
        from_im = (
            from_im.convert("RGB") if self.opts.label_nc == 0 else from_im.convert("L")
        )

        to_path = self.target_paths[index]
        to_im = Image.open(to_path).convert("RGB")
        if self.target_transform:
            to_im = self.target_transform(to_im)

        if self.source_transform:
            from_im = self.source_transform(from_im)
        else:
            from_im = to_im

        if self.opts.augmentation:
            _, origin_h, origin_w = from_im.shape
            PAD_MAX = origin_h // 2

            # Flip
            if random() < 0.5:
                from_im = torch.flip(from_im, [2])
                to_im = torch.flip(to_im, [2])

            crop_i, crop_j, crop_h, crop_w = transforms.RandomResizedCrop.get_params(
                from_im, scale=(0.5, 1), ratio=(0.75, 1.25)
            )
            pad_arr = [math.floor(random() * PAD_MAX) for _ in range(4)]
            angle = random() * 90 - 45

            # Crop or Pad
            if random() < 0.5:  # Do crop
                from_im = F.crop(from_im, crop_i, crop_j, crop_h, crop_w)
                to_im = F.crop(to_im, crop_i, crop_j, crop_h, crop_w)
            else:  # Do pad
                from_im = F.pad(from_im, pad_arr)
                to_im = F.pad(to_im, pad_arr)

            # Rotate and Resize
            from_im = F.rotate(from_im, angle)
            from_im = F.resize(from_im, (origin_h, origin_w), antialias=True)
            to_im = F.rotate(to_im, angle)
            to_im = F.resize(to_im, (origin_h, origin_w), antialias=True)

        return from_im, to_im
