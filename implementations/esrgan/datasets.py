import glob
import random
import os
import numpy as np



import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import imgproc
import cv2

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        hr_height, hr_width = hr_shape
        self.hr_height, self.hr_width = hr_shape
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
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))


    def __getitem__(self, index: int):
        # Read a batch of image data
        # image = cv2.imread(self.files[index % len(self.files)], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img = Image.open(self.files[index % len(self.files)])
        # Image processing operations


        # hr_image = imgproc.random_crop(image, self.image_size)
        # hr_image = cv2.random_crop(image, (self.image_size, self.image_size))
        hr_image = img.resize((self.hr_height, self.hr_height), resample=Image.BICUBIC)

        lr_image = hr_image.resize((self.hr_height//4, self.hr_height//4), resample=Image.BICUBIC)
        # lr_image = lr_image.resize((self.hr_height, self.hr_height), resample=Image.BICUBIC)


        # Only extract the image data of the Y channel
        # lr_y_image = imgproc.bgr2ycbcr(lr_image, only_use_y_channel=True)
        # hr_y_image = imgproc.bgr2ycbcr(hr_image, only_use_y_channel=True)

        # lr_y_image = cv2.cvtColor(lr_image, cv2.COLOR_YCrCb2BGR)[:,:,0]
        # hr_y_image = cv2.cvtColor(hr_image, cv2.COLOR_YCrCb2BGR)[:,:,0]

        hr_y_image, _, _ = hr_image.convert('YCbCr').split()
        lr_y_image, _, _ = lr_image.convert('YCbCr').split()


        hr_y_image = np.array(hr_y_image)
        lr_y_image = np.array(lr_y_image)

        # print(hr_y_image.shape, '!!!!!!!!!!!!!!!')
        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        # lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=False)
        # hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=False)

        lr_y_tensor = torch.from_numpy(lr_y_image).unsqueeze(0)
        hr_y_tensor = torch.from_numpy(hr_y_image).unsqueeze(0)


        return {"lr": lr_y_tensor, "hr": hr_y_tensor}

    # def __getitem__(self, index):
    #     img = Image.open(self.files[index % len(self.files)])
    #     img_lr = self.lr_transform(img)
    #     img_hr = self.hr_transform(img)

    #     return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)
