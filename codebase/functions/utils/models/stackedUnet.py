

import torch
import torch.nn as nn
import torch.nn.functional as F

# Correction Multi-input model
class StackedUnet(nn.Module):
    '''
    Pytorch implementation of the stacked U-net from 
    Al-masni et al. "Stacked U-Nets with self-assisted priors towards 
    robust correction of rigid motion artifact in brain MRI", in NeuroImage, 2022.
    '''
    def __init__(self, unet_num_ch_first_layer=32, norm_type='batch'):
        super(StackedUnet, self).__init__()

        if norm_type == 'batch':
            self.norm_layer = nn.BatchNorm2d
        elif norm_type == 'instance':
            self.norm_layer = nn.InstanceNorm2d
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
        
        self.conv1 = self.conv_layer(1, 32)
        self.conv2 = self.conv_layer(1, 32)
        self.conv3 = self.conv_layer(1, 32)
        self.unet1 = UNet(num_input_channels=96, unet_num_ch_first_layer=unet_num_ch_first_layer, norm_layer=self.norm_layer)
        self.unet2 = UNet(num_input_channels=97, unet_num_ch_first_layer=unet_num_ch_first_layer, norm_layer=self.norm_layer)

    def conv_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            self.norm_layer(out_ch),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3):
        # Input Convolution for 3 inputs
        conv1 = self.conv1(x1)
        conv2 = self.conv2(x2)
        conv3 = self.conv3(x3)


        # First UNet Prediction
        pred_1 = self.unet1(torch.cat([conv1, conv2, conv3], dim=1))

        # Second UNet Prediction with concatenation
        input_concat = torch.cat([conv1, conv2, conv3, pred_1], dim=1)
        pred_2 = self.unet2(input_concat)

        return pred_2

# CBAM --------------------------------------------
# Convolutional Block Attention Module (CBAM) block
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        # in_planes: number of input channels
        super(ChannelAttention, self).__init__()
        # For each channel we obtina the a single average and max value
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Two fully connected layers implement as 1x1 conv layers
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=True)
        
        #self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = self.fc2(F.relu(self.fc1(avg_out)))
        max_out = self.fc2(F.relu(self.fc1(max_out)))
        out = avg_out + max_out
        return torch.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        #assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        #padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Take mean and max across the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# UNet Architecture with CBAM
class UNet(nn.Module):
    def __init__(self, num_input_channels, unet_num_ch_first_layer, norm_layer):
        super(UNet, self).__init__()
        
        self.k1 = unet_num_ch_first_layer
        self.k2 = unet_num_ch_first_layer * 2
        self.k3 = unet_num_ch_first_layer * 4
        self.k4 = unet_num_ch_first_layer * 8

        # Contracting path
        self.conv1 = self.conv_block(num_input_channels, self.k1, norm_layer)
        self.conv2 = self.conv_block(self.k1, self.k2, norm_layer)
        self.conv3 = self.conv_block(self.k2, self.k3, norm_layer)
        self.conv4 = self.conv_block(self.k3, self.k4, norm_layer)

        # Expansive path
        self.deconv1 = self.deconv_block(self.k4+self.k3, self.k3, norm_layer)
        self.deconv2 = self.deconv_block(self.k3+self.k2, self.k2, norm_layer)
        self.deconv3 = self.deconv_block(self.k2+self.k1, self.k1, norm_layer)

        # CBAM blocks
        self.cbam1 = CBAM(self.k1)
        self.cbam2 = CBAM(self.k2)
        self.cbam3 = CBAM(self.k3)
        self.cbam4 = CBAM(self.k4)
        self.cbam5 = CBAM(self.k3)
        self.cbam6 = CBAM(self.k2)
        self.cbam7 = CBAM(self.k1)

        # Final output
        self.final_conv = nn.Conv2d(self.k1, 1, kernel_size=3, padding=1)

    def conv_block(self, in_ch, out_ch, norm_layer):
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_ch),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_ch),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
        )
        return block

    def deconv_block(self, in_ch, out_ch, norm_layer):
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_ch),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_ch),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
        )
        return block

    def forward(self, c1_cbam):
        # Contracting Path
        c1_cbam = self.conv1(c1_cbam)
        c1_cbam = self.cbam1(c1_cbam)
        c2_cbam = F.avg_pool2d(c1_cbam, kernel_size=2) # downsample

        c2_cbam = self.conv2(c2_cbam)
        c2_cbam = self.cbam2(c2_cbam)
        c3_cbam = F.avg_pool2d(c2_cbam, kernel_size=2) # downsample

        c3_cbam = self.conv3(c3_cbam)
        c3_cbam = self.cbam3(c3_cbam)
        c4_cbam = F.avg_pool2d(c3_cbam, kernel_size=2) # downsample

        # Transition Layer
        c4_cbam = self.conv4(c4_cbam)
        c4_cbam = self.cbam4(c4_cbam)

        # Expansive Path
        c4_cbam = F.interpolate(c4_cbam, scale_factor=2, mode='nearest')

        # reflect pad on the right/botton if needed to handle odd input dimensions
        padding = [0, 0, 0, 0]
        if c4_cbam.shape[-1] != c3_cbam.shape[-1]:
            padding[1] = 1  # padding right
        if c4_cbam.shape[-2] != c3_cbam.shape[-2]:
            padding[3] = 1  # padding bottom
        if torch.sum(torch.tensor(padding)) != 0:
            c4_cbam = F.pad(c4_cbam, padding, "reflect")

        c3_cbam = torch.cat([c4_cbam, c3_cbam], dim=1)
        c3_cbam = self.deconv1(c3_cbam)

        c3_cbam = self.cbam5(c3_cbam)

        c3_cbam = F.interpolate(c3_cbam, scale_factor=2, mode='nearest')

        # reflect pad on the right/botton if needed to handle odd input dimensions
        padding = [0, 0, 0, 0]
        if c3_cbam.shape[-1] != c2_cbam.shape[-1]:
            padding[1] = 1  # padding right
        if c3_cbam.shape[-2] != c2_cbam.shape[-2]:
            padding[3] = 1  # padding bottom
        if torch.sum(torch.tensor(padding)) != 0:
            c3_cbam = F.pad(c3_cbam, padding, "reflect")

        c2_cbam = torch.cat([c3_cbam, c2_cbam], dim=1)
        c2_cbam = self.deconv2(c2_cbam)

        c2_cbam = self.cbam6(c2_cbam)

        c2_cbam = F.interpolate(c2_cbam, scale_factor=2, mode='nearest')

        # reflect pad on the right/botton if needed to handle odd input dimensions
        padding = [0, 0, 0, 0]
        if c2_cbam.shape[-1] != c1_cbam.shape[-1]:
            padding[1] = 1  # padding right
        if c2_cbam.shape[-2] != c1_cbam.shape[-2]:
            padding[3] = 1  # padding bottom
        if torch.sum(torch.tensor(padding)) != 0:
            c2_cbam = F.pad(c2_cbam, padding, "reflect")

        c1_cbam = torch.cat([c2_cbam, c1_cbam], dim=1)
        c1_cbam = self.deconv3(c1_cbam)

        c1_cbam = self.cbam7(c1_cbam)

        c1_cbam = self.final_conv(c1_cbam)
        return c1_cbam



# Example usage
if __name__ == '__main__':
    input_height, input_width = 130, 130  # Set your input size
    model = StackedUnet()
    x1 = torch.randn(1, 1, input_height, input_width)
    x2 = torch.randn(1, 1, input_height, input_width)
    x3 = torch.randn(1, 1, input_height, input_width)
    
    output = model(x1, x2, x3)
    print(output.shape)