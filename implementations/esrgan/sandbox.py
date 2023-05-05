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



def rgb_to_ycrcb(img):
    r, g, b = img[0], img[1], img[2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128
    cb = (b - y) * 0.564 + 128
    ycrcb = torch.stack([y, cr, cb], dim=0)
    return ycrcb



# def ycrcb_to_rgb(img):
#     y, cr, cb = img[0], img[1], img[2]
#     r = y + 1.402 * (cr - 128)
#     g = y - 0.344 * (cb - 128) - 0.714 * (cr - 128)
#     b = y + 1.772 * (cb - 128)
#     rgb = torch.stack([r, g, b], dim=0)
#     return rgb


def ycrcb_to_rgb(img):
    r"""Converts a YCrCb image tensor to an RGB image tensor.
    Args:
        img (torch.Tensor): Input image tensor with dimensions (B, 3, H, W).
    Returns:
        torch.Tensor: RGB image tensor with dimensions (B, 3, H, W).
    """
    y, cr, cb = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
    r = y + 1.402 * (cr - 128)
    g = y - 0.344 * (cb - 128) - 0.714 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return torch.stack([r, g, b], dim=1)

def ycrcb_to_rgb_single(img):
    r"""Converts a YCrCb image tensor to an RGB image tensor.
    Args:
        img (torch.Tensor): Input image tensor with dimensions (B, 3, H, W).
    Returns:
        torch.Tensor: RGB image tensor with dimensions (B, 3, H, W).
    """
    y, cr, cb = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]
    r = y + 1.402 * (cr - 128)
    g = y - 0.344 * (cb - 128) - 0.714 * (cr - 128)
    b = y + 1.772 * (cb - 128)
    return torch.stack([r, g, b], dim=1)



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
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        # save_image(img_hr, "ycrcb_hr.png")

        print(torch.allclose(img_hr, ycrcb_to_rgb(rgb_to_ycrcb(img_hr)), atol=1e-5, rtol=1e-5), "!!!!!!!!!!!")
        print(img_hr[0].numpy(), "AAAA")
        print(ycrcb_to_rgb(rgb_to_ycrcb(img_hr))[0].numpy()/255.0, "BBBB")

        ycrcb_hr = rgb_to_ycrcb(img_hr)  # extract Y channel

        ycrcb_lr = rgb_to_ycrcb(img_lr) # extract Y channel
        
        rgb_hr = ycrcb_to_rgb(ycrcb_hr)


        return {"lr": ycrcb_lr[0].unsqueeze(0), "hr": ycrcb_hr[0].unsqueeze(0), "ycrcb_lr":ycrcb_lr, "ycrcb_hr": ycrcb_hr}

    def __len__(self):
        return len(self.files)

hr_height = 256
hr_transform = transforms.Compose(
    [
        transforms.Resize((hr_height, hr_height), Image.BICUBIC),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ]
)

lr_transform_sub = transforms.Compose(
    [
        transforms.Resize((hr_height, hr_height), Image.BICUBIC)
    ]
)
lr_transform = transforms.Compose(
    [
        transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ]
)


img = Image.open("/home/ubuntu/PyTorch-GAN/data/BSR/BSDS500/data/images/train/2092.jpg")
img_hr = hr_transform(img)
img_lr = lr_transform(img)


img_hr_sub = rgb_to_ycrcb(lr_transform_sub(img_lr))


ycrcb = rgb_to_ycrcb(img_lr)
y_channel_output = lr_transform_sub(ycrcb[0].unsqueeze(0))

print(y_channel_output.shape, "@@@")
save_image(y_channel_output, "y_channel.png")

img_hr_sub[0] = y_channel_output

rgb = ycrcb_to_rgb(img_hr_sub.unsqueeze(0))

rgb = denormalize(rgb)






save_image(img_hr, "input.png")
# save_image(y_ch, "output.png")


