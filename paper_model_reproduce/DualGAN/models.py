"""""
from <Inter‑site harmonization based on dual generative adversarial networks 
    for diffusion tensor imaging application to neonatal white matter development>
    
    :A common choice is to keep the kernel size at 3x3 or 5x5. The first convolutional layer is often kept larger.
    :Its size is less important as there is only one first layer, and it has fewer input channels: 3, 1 by color.
"""""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(ConvBlock, self).__init__()

        # Conv:k5s2cN + LReLU + BN
        conv_block = [nn.Conv2d(in_nc, out_nc, kernel_size=5, stride=2, padding=2, padding_mode="reflect"),
                      nn.BatchNorm2d(out_nc),
                      nn.LeakyReLU(inplace=True)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(DeconvBlock, self).__init__()

        # Deconv:k5s1/2cN + LReLU +BN
        deconv_block = [nn.ConvTranspose2d(in_nc, out_nc, kernel_size=5, stride=2, padding=2),
                        nn.BatchNorm2d(out_nc),
                        nn.ReLU(inplace=True)]

        self.deconv_block = nn.Sequential(*deconv_block)

    def forward(self, x):
        return self.deconv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=5, stride=2)

        # Downsampling
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.conv5 = ConvBlock(512, 512)
        self.conv6 = ConvBlock(512, 512)
        self.conv7 = ConvBlock(512, 512)
        self.conv8 = ConvBlock(512, 512)

        # Upsampling
        self.deconv9 = DeconvBlock(512, 512)
        self.deconv10 = DeconvBlock(512, 512)
        self.deconv11 = DeconvBlock(512, 512)
        deconv12_block = [nn.ConvTranspose2d(512, 512, kernel_size=5, stride=2, padding=2, output_padding=1),
                          nn.BatchNorm2d(512),
                          nn.ReLU(inplace=True)]
        self.deconv12 = nn.Sequential(*deconv12_block)
        self.deconv13 = DeconvBlock(512, 256)
        self.deconv14 = DeconvBlock(256, 128)
        deconv15_block = [nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
                          nn.BatchNorm2d(64),
                          nn.ReLU(inplace=True)]
        self.deconv15 = nn.Sequential(*deconv15_block)

        # Output layer
        self.deconv16 = nn.ConvTranspose2d(64, output_nc, kernel_size=5, stride=2, output_padding=1)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        c7 = self.conv7(c6)
        c8 = self.conv8(c7)
        dc9 = self.deconv9(c8)
        dc10 = self.deconv10(c7 + dc9)
        dc11 = self.deconv11(c6 + dc10)
        dc12 = self.deconv12(c5 + dc11)
        dc13 = self.deconv13(c4 + dc12)
        dc14 = self.deconv14(c3 + dc13)
        dc15 = self.deconv15(c2 + dc14)
        dc16 = self.deconv16(c1 + dc15)
        return dc16


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(256, 1, 3, stride=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


if __name__ == "__main__":
    x = torch.rand(size=(1, 1, 280, 280))
    generator = Generator(1, 1)
    fake_img = generator(x)
    print(fake_img.size())
    discriminator = Discriminator(1)
    out = discriminator(fake_img)
    print(out.size())
    print(out)
