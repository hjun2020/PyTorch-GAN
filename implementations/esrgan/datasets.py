import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from torchvision.utils import save_image


# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# def rgb_to_ycrcb(img):
#     # make sure the input is a 4D tensor with dimensions (batch_size, channels, height, width)
#     assert len(img.shape) == 4, "Input tensor must be 4D"
#     assert img.shape[1] == 3, "Input tensor must have 3 channels"
    
#     # convert the image to YCrCb color space
#     ycrcb_img = torch.zeros_like(img)
#     for i in range(img.shape[0]):
#         ycrcb_img[i] = torch.Tensor([0.299, 0.587, 0.114, -0.168736, -0.331264, 0.5, 0.5, -0.418688, -0.081312]).view(3,3).mm(img[i].view(3,-1)).view(3, img.shape[2], img.shape[3])
    
#     # extract the Y channel
#     y_channel = ycrcb_img[:, 0:1, :, :]
    
#     return y_channel

def rgb_to_ycrcb(img):
    r, g, b = img[0], img[1], img[2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128
    cb = (b - y) * 0.564 + 128
    ycrcb = torch.stack([y, cr, cb], dim=0)
    return ycrcb

# def ycrcb_to_rgb(img):
#     r"""Converts a YCrCb image tensor to an RGB image tensor.
#     Args:
#         img (torch.Tensor): Input image tensor with dimensions (B, 3, H, W).
#     Returns:
#         torch.Tensor: RGB image tensor with dimensions (B, 3, H, W).
#     """
#     y, cr, cb = torch.chunk(img, chunks=3, dim=1)
#     r = y + 1.402 * (cr - 0.5)
#     g = y - 0.344 * (cb - 0.5) - 0.714 * (cr - 0.5)
#     b = y + 1.772 * (cb - 0.5)
#     return torch.cat([r, g, b], dim=1)

def ycrcb_to_rgb(img):
    y, cr, cb = img[0], img[1], img[2]
    r = y + 1.402 * (cr - 0.5)
    g = y - 0.344 * (cb - 0.5) - 0.714 * (cr - 0.5)
    b = y + 1.772 * (cb - 0.5)
    rgb = torch.stack([r, g, b], dim=0)
    return rgb

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )

        self.lr_transform_sub = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC)
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)


        ycrcb_hr = rgb_to_ycrcb(img_hr)  # extract Y channel
        ycrcb_lr = rgb_to_ycrcb(img_lr) # extract Y channel

        img_hr_sub = rgb_to_ycrcb(self.lr_transform_sub(img_lr))

        


        return {"lr": ycrcb_lr[0].unsqueeze(0), "hr": ycrcb_hr[0].unsqueeze(0), "target_hr":img_hr, "img_hr_sub": img_hr_sub}

    def __len__(self):
        return len(self.files)





