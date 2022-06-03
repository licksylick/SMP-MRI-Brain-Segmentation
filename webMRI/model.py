import torch
import torch.nn as nn

# convolution block
def conv_block(in_channel, feature):
    block = [nn.Conv2d(in_channel, feature, kernel_size=3, padding=1),
             nn.BatchNorm2d(feature),
             nn.ReLU(),
             nn.Conv2d(feature, feature, kernel_size=3, padding=1),
             nn.BatchNorm2d(feature),
             nn.ReLU()]

    return nn.Sequential(*block)


# U-net
class Unet(nn.Module):
    def __init__(self, in_channels=3, features=32, out_channels=1):
        init_features = features
        super(Unet, self).__init__()
        self.enc1 = conv_block(in_channels, features)  # output 1, 32, 256, 256
        self.pool1 = nn.MaxPool2d(2, 2)  # output 1, 32, 128, 128
        self.enc2 = conv_block(features, features * 2)  # output 1, 64, 128, 128
        self.pool2 = nn.MaxPool2d(2, 2)  # output 1, 64, 64, 64
        self.enc3 = conv_block(features * 2, features * 4)  # output 1, 128, 64, 64
        self.pool3 = nn.MaxPool2d(2, 2)  # output 1, 128, 32, 32
        self.enc4 = conv_block(features * 4, features * 8)  # output 1, 256, 32, 32
        self.pool4 = nn.MaxPool2d(2, 2)  # output 1, 256, 16, 16

        self.bottleneck = nn.Conv2d(features * 8, features * 16, kernel_size=3, padding=1)  # output 1, 512, 16, 16

        # transposed convolution is used for segmentation, super resolution
        # deconvolution is used for image reconstruction

        self.up4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)  # output 1, 256, 32, 32
        self.dcd4 = conv_block(features * 8 * 2, features * 8)  # from concatenation, output 1, 256, 32, 32
        self.up3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)  # output 1, 128, 64, 64
        self.dcd3 = conv_block(features * 4 * 2, features * 4)  # from concatenation, output 1, 128, 64, 64
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)  # output 1, 64, 128, 128
        self.dcd2 = conv_block(features * 2 * 2, features * 2)  # from concatenation, output 1, 64, 128, 128
        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)  # output 1, 32, 256, 256
        self.dcd1 = conv_block(features * 2, features)  # from concatenation, output 1, 32, 256, 256

        self.final = nn.Conv2d(features, out_channels, kernel_size=1)  # output 1, 1, 256, 256

    def forward(self, x):
        enc1 = self.enc1(x)  # output 1, 32, 256, 256
        enc2 = self.enc2(self.pool1(enc1))  # output 1, 64, 128, 128
        enc3 = self.enc3(self.pool2(enc2))  # output 1, 128, 64, 64
        enc4 = self.enc4(self.pool3(enc3))  # output 1, 256, 32, 32
        bottleneck = self.bottleneck(self.pool4(enc4))  # output 1, 512, 16, 16
        dcd4 = self.up4(bottleneck)  # output 1, 256, 32, 32
        dcd4 = self.dcd4(torch.cat((dcd4, enc4), dim=1))  # output 1, 256, 32, 32
        dcd3 = self.up3(dcd4)  # output 1, 128, 64, 64
        dcd3 = self.dcd3(torch.cat((dcd3, enc3), dim=1))  # output 1, 128, 64, 64
        dcd2 = self.up2(dcd3)  # output 1, 64, 128, 128
        dcd2 = self.dcd2(torch.cat((dcd2, enc2), dim=1))  # output 1, 64, 128, 128
        dcd1 = self.up1(dcd2)  # output 1, 32, 256, 256
        dcd1 = self.dcd1(torch.cat((dcd1, enc1), dim=1))  # output 1, 32, 256, 256
        return torch.sigmoid(self.final(dcd1))

