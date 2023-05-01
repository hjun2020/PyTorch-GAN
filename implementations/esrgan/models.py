import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math





class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)


# class DenseResidualBlock(nn.Module):
#     """
#     The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
#     """

#     def __init__(self, filters, res_scale=0.2):
#         super(DenseResidualBlock, self).__init__()
#         self.res_scale = res_scale

#         def block(in_features, non_linearity=True):
#             layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
#             if non_linearity:
#                 layers += [nn.LeakyReLU()]
#             return nn.Sequential(*layers)

#         self.b1 = block(in_features=1 * filters)
#         self.b2 = block(in_features=2 * filters)
#         self.b3 = block(in_features=3 * filters)
#         self.b4 = block(in_features=4 * filters)
#         self.b5 = block(in_features=5 * filters, non_linearity=False)
#         self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

#     def forward(self, x):
#         inputs = x
#         for block in self.blocks:
#             out = block(inputs)
#             inputs = torch.cat([inputs, out], 1)
#         return out.mul(self.res_scale) + x


# class ResidualInResidualDenseBlock(nn.Module):
#     def __init__(self, filters, res_scale=0.2):
#         super(ResidualInResidualDenseBlock, self).__init__()
#         self.res_scale = res_scale
#         self.dense_blocks = nn.Sequential(
#             DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
#         )

#     def forward(self, x):
#         return self.dense_blocks(x).mul(self.res_scale) + x


# class GeneratorRRDB(nn.Module):
#     def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
#         super(GeneratorRRDB, self).__init__()

#         # First layer
#         self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
#         # Residual blocks
#         self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
#         # Second conv layer post residual blocks
#         self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
#         # Upsampling layers
#         upsample_layers = []
#         for _ in range(num_upsample):
#             upsample_layers += [
#                 nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
#                 nn.LeakyReLU(),
#                 nn.PixelShuffle(upscale_factor=2),
#             ]
#         self.upsampling = nn.Sequential(*upsample_layers)
#         # Final output block
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
#         )

#     def forward(self, x):
#         out1 = self.conv1(x)
#         out = self.res_blocks(out1)
#         out2 = self.conv2(out)
#         out = torch.add(out1, out2)
#         out = self.upsampling(out)
#         out = self.conv3(out)
#         return out
    






##########################################################################################################

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out

class GeneratorRRDB(nn.Module):
    ################### TODO######################################
    ## The channels must be from an argument in esrgan.py args!!!!!!!!!!!!!!!!!!!!!!
    def __init__(self, channels=1, filters=64, num_upsample=2, num_res_blocks=16):
        super(GeneratorRRDB, self).__init__()
        self.scale_factor = num_upsample * 2
        self.num_residual_blocks = num_res_blocks

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=9, padding=4)
        self.prelu1 = nn.PReLU()
        
        # Add residual blocks
        res_blocks = []
        for i in range(self.num_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Add upsampling layers
        upsampling = []
        for _ in range(int(self.scale_factor/2)):
            upsampling += [
                nn.Conv2d(64, 256, kernel_size=3, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                # nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),

                nn.PReLU()
            ]
        self.upsampling = nn.Sequential(*upsampling)
        
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        
        residual = x
        
        x = self.res_blocks(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x += residual
        
        x = self.upsampling(x)
        
        x = self.conv3(x)
        
        return x



##########################################################################################################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
