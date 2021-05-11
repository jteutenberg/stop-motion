import torch
import torch.nn as nn
import torch.nn.functional as F

def unet_block(in_channels, out_channels, hold_in=False):
    """ 
    Create a (convolution(3) + batch norm + relu ) x2 
    Channel numbers are altered between the two sub-blocks
    """ 
    mid_channels = out_channels
    if hold_in:
        mid_channels = in_channels
    return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

def unet_down(in_channels, out_channels):
    """ Downsample via a max pool and standard unet block """
    return nn.Sequential(
            nn.MaxPool2d(2), # half size
            unet_block(in_channels, out_channels)
            )

def unet_up(in_channels, out_channels):
    """ Upsample via a pixel shuffle and a standard unet block """
    return nn.Sequential(

            )
class Up(nn.Module):
    """
    Upscales via a pixel shuffle, then a standard unet block
    After the pixel shuffle this also pulls in output from an earlier down block
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.PixelShuffle(2) # and will have 1/4 the channels
        # we'll also have half as many input channels moving "across" as compared to up
        self.conv = unet_block(in_channels//4 + in_channels//2, out_channels, True)


    def forward(self, x1, x2):
        """ Output based on x1 (from lower block) and x2 (earlier up block) """
        # pixel shuffle up from lower block
        x1 = self.up(x1)
        # ensure other input matches dimensions (could have lost border pixels)
        #diffY = x2.size()[2] - x1.size()[2]
        #diffX = x2.size()[3] - x1.size()[3]
        #x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                diffY // 2, diffY - diffY // 2])
        # concatenate the inputs
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class InterpolationUNet(nn.Module):
    def __init__(self):
        super(InterpolationUNet, self).__init__()
        self.in_channels = 6 # two input images of three channels each
        self.out_channels = 3 # one output image

        self.input_layer = unet_block(self.in_channels, 32) # 3x3 features, expand to 32 channels
        self.down1 = unet_down(32, 64) # halve size, double channels 
        self.down2 = unet_down(64, 128)
        self.down3 = unet_down(128, 256) # 81x81 features
        # the first upward step will take the input and output of the last down step
        self.up1 = Up(256, 128) # will have 256+128 input channels
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.output_layer = nn.Conv2d(32, self.out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.input_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.output_layer(x)
        return x

