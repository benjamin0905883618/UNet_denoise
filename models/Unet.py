from .Unet_part import *

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear = False):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.input = DoubleConv(n_channels, 64)
        self.down1 = DownSampling(64, 128)
        self.down2 = DownSampling(128, 256)
        self.down3 = DownSampling(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownSampling(512, 1024)
        self.up1 = UpSampling(1024, 512//factor, bilinear)
        self.up2 = UpSampling(512, 256//factor, bilinear)
        self.up3 = UpSampling(256, 128//factor, bilinear)
        self.up4 = UpSampling(128, 64//factor, bilinear)
        self.out = OutConv(64, n_classes)
    def forward(self, x):
        x0 = self.input(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        x = self.out(x)
        return x
