import torch.nn as nn
import torch.nn.functional as F
import torch


class ResBlock_Down(nn.Module):
    """
    Implementation of Residual block for downsampling part of Unet.

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_dropout=False, first_layer=False):
        super(ResBlock_Down, self).__init__()
        # First layer is different than other layers
        self.is_first_layer = first_layer

        if self.is_first_layer:
            self.first_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
            )
        else:
            self.res_block = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
        
        # Residual path
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, input):
        res = self.conv1x1(input)

        if self.is_first_layer:
            x = self.first_block(input)
        else:
            x = self.res_block(input)
        x += res

        return x
    

class ResBlock_Up(nn.Module):
    """
    Implementation of Residual block for upsampling part of Unet.
    """

    def __init__(self, cancated_channels, out_channels, upsample_mode='bilinear'):
        super(ResBlock_Up, self).__init__()

        self.up_samples = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=True)

        self.res_block = nn.Sequential(
            nn.BatchNorm2d(cancated_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cancated_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.conv1x1 = nn.Conv2d(cancated_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, input, skip_input):
        x = self.up_samples(input)
        x = torch.cat([x, skip_input], dim=1)

        res = self.conv1x1(x)
        x = self.res_block(x)

        x += res

        return x


class Generator(nn.Module):
    """
    Implementation of Residual UNet architecture as the generator
    """

    def __init__(self, in_channels=3, out_channels=3, downsample_rate=8):
        """
        Initializing UNet architecture for the generator

        Args:
                in_channels: The number of the input channels (3 for RGB images).
                out_channels: The number of the output channels (3 since the output is an image).
                downsample_rate: output stride, can be 8 or 16. Factor of that the input image is downsampled at the
                                 coarsest resolution.
                                 e.g input: 256 x 256
                                     downsample_rate: 8
                                     256 / 8 = 32 x 32 feature map at coarsest resolution.

        Example to initialize the network:
            network = Res_UNet(in_channels=3, out_channels=1, downsample_rate=16)
        """

        super(Generator, self).__init__()
        self.downsample_rate = downsample_rate
        assert downsample_rate == 8 or downsample_rate == 16

        # Layers
        self.encoder0 = ResBlock_Down(in_channels, 64, first_layer=True)
        self.encoder1 = ResBlock_Down(64, 128, stride=2)
        self.encoder2 = ResBlock_Down(128, 256, stride=2)

        if downsample_rate == 8:
            self.bridge = ResBlock_Down(256, 512, stride=2)

        if downsample_rate == 16:
            self.encoder3 = ResBlock_Down(256, 512, stride=2)
            self.bridge = ResBlock_Down(512, 512, stride=2)
            # concated_input size is sum of bridge and encoder3 output channels
            self.decoder0 = ResBlock_Up(1024, 512)
        
        # concated_input size is sum of bridge (or decoder0) and encoder2 output channels
        self.decoder1 = ResBlock_Up(768, 256)
        # concated_input size is sum of decoder1 and encoder1 output channels
        self.decoder2 = ResBlock_Up(384, 128)
        # concated_input size is sum of decoder2 and encoder0 output channels
        self.decoder3 = ResBlock_Up(192, 64)

        self.outConv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.tanh_act = nn.Tanh()

        # Initialize weights of the generator network
        self.apply(self.init_weights)

    def forward(self, input):
        x0 = self.encoder0(input)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)

        if self.downsample_rate==8:
            x = self.bridge(x2)

        if self.downsample_rate==16:
            x3 = self.encoder3(x2)
            x = self.bridge(x3)
            x = self.decoder0(x, x3)
        
        x = self.decoder1(x, x2)
        x = self.decoder2(x, x1)
        x = self.decoder3(x, x0)
        x = self.outConv(x)

        return self.tanh_act(x)
    
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.constant_(m.bias, 0)
    

class Discriminator(nn.Module):
    """
    Implementing a simple Neural Network as a discriminator.
        The network has 5 blocks of ResNet and 2 linear layers.
    """
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            ResBlock_Down(in_channels, 64, first_layer=True),
            ResBlock_Down(64, 128, stride=2),
            ResBlock_Down(128, 256, stride=2),
            ResBlock_Down(256, 512, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=3, stride=1)
        )

        # Initialize weights of the discriminator network
        self.apply(self.init_weights)
            

    def forward(self, input_image):
        
        return self.discriminator(input_image)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.constant_(m.bias, 0)