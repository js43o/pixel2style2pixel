import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
from random import random
from torchvision.transforms import transforms
import torchvision.transforms.functional as F


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
            should_flip, should_crop = random() < 0.5, random() < 0.5
            if should_flip:
                from_im = torch.flip(from_im, [2])
                to_im = torch.flip(to_im, [2])

            if should_crop:
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    from_im, scale=(0.1, 1), ratio=(0.5, 2)
                )
                _, origin_h, origin_w = from_im.shape
                resize = transforms.Resize((origin_h, origin_w), antialias=True)
                from_im = resize(F.crop(from_im, i, j, h, w))
                to_im = resize(F.crop(to_im, i, j, h, w))

        return from_im, to_im
