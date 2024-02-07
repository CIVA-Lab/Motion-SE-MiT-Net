import torch
import torch.nn as nn
import torchvision

from torchvision import models

# double conv + bn + relu
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True)
    )

# MUMiTNet network model
class MUMiTNet(nn.Module):

    def __init__(self, n_out_class, mvt_encoder):
        super().__init__()
        
        self.mvt_encoder = mvt_encoder
     
        # ******************** Encoder part Second Stream  **********************
        # get pretrained resnet 18
        rnet_base_model = models.resnet18(pretrained=True)
        self.rnet_base_layers = list(rnet_base_model.children())  

        # Layer 0 (Resnet 50) 
        self.layer_0_2 = nn.Sequential(*self.rnet_base_layers[:3])
        
        # Layer 1 (Resnet 50) 
        self.layer_1_2 = nn.Sequential(*self.rnet_base_layers[3:5])
        
        # Layer 2 (Resnet 50) 
        self.layer_2_2 = self.rnet_base_layers[5]
        
        # Layer 3 (Resnet 50) 
        self.layer_3_2 = self.rnet_base_layers[6]
        
        # Layer 4 (Resnet 50) 
        self.layer_4_2 = self.rnet_base_layers[7]
        # *********************************************************
        
        # ******************** Decoder part  **********************
        # define upsampling
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # define  double convolution after each upsample
        # input channels defined by previous upsample channels + skip connection channels
        self.conv_d_4 = double_conv(320 + 256 + 512 + 512, 256)
        # input channels defined by previous upsample channels + skip connection channels
        self.conv_d_3 = double_conv(128 + 128 + 256, 128)
        # input channels defined by previous upsample channels + skip connection channels
        self.conv_d_2 = double_conv(64 + 64 + 128, 64)
        # input channels defined by previous upsample channels + skip connection channels
        self.conv_d_1 = double_conv(64 + 64, 32)
        # input channels defined by previous upsample channels + skip connection channels
        self.conv_d_0 = double_conv(32, 16)

        # **********************************************************
        
        # final convolution later
        self.conv_final = nn.Conv2d(16, n_out_class, 3, padding=1)
        
    def forward(self, input, input2):
    
        features = self.mvt_encoder(input)
        
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]
        
        layer_0_2 = self.layer_0_2(input2)
        layer_1_2 = self.layer_1_2(layer_0_2)
        layer_2_2 = self.layer_2_2(layer_1_2)
        layer_3_2 = self.layer_3_2(layer_2_2)
        layer_4_2 = self.layer_4_2(layer_3_2)

        # decoder part
        d_out = torch.cat([head, layer_4_2], dim=1)
        d_out = self.up_sample(d_out)

        d_out = torch.cat([d_out, skips[0], layer_3_2], dim=1)
        d_out = self.conv_d_4(d_out)
 
        d_out = self.up_sample(d_out)
        d_out = torch.cat([d_out, skips[1], layer_2_2], dim=1)
        d_out = self.conv_d_3(d_out)

        d_out = self.up_sample(d_out)
        d_out = torch.cat([d_out, skips[2], layer_1_2], dim=1)
        d_out = self.conv_d_2(d_out)

        d_out = self.up_sample(d_out)
        d_out = torch.cat([d_out, skips[3], layer_0_2], dim=1)
        d_out = self.conv_d_1(d_out)
        
        d_out = self.up_sample(d_out)
        d_out = self.conv_d_0(d_out)   
        
        # final convolution later
        out = self.conv_final(d_out)        
        
        return out
