import torch
import torch.nn as nn

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

# UMiTNet network model
class UMiTNet(nn.Module):

    def __init__(self, n_out_class, mvt_encoder):
        super().__init__()
        
        self.mvt_encoder = mvt_encoder
 
        # ******************** Decoder part  **********************
        # define upsampling
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # define  double convolution after each upsample
        # input channels defined by previous upsample channels + skip connection channels
        self.conv_d_4 = double_conv(320 + 512, 256)
        # input channels defined by previous upsample channels + skip connection channels
        self.conv_d_3 = double_conv(128  + 256, 128)
        # input channels defined by previous upsample channels + skip connection channels
        self.conv_d_2 = double_conv(64 + 128, 64)
        # input channels defined by previous upsample channels + skip connection channels
        self.conv_d_1 = double_conv(64, 32)
        # input channels defined by previous upsample channels + skip connection channels
        self.conv_d_0 = double_conv(32, 16)

        # **********************************************************
        
        # final convolution later
        self.conv_final = nn.Conv2d(16, n_out_class, 3, padding=1)
        
    def forward(self, input):
    
        features = self.mvt_encoder(input)
        
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        # decoder part
        d_out = self.up_sample(head)
        d_out = torch.cat([d_out, skips[0]], dim=1)
        d_out = self.conv_d_4(d_out)
 
        d_out = self.up_sample(d_out)
        d_out = torch.cat([d_out, skips[1]], dim=1)
        d_out = self.conv_d_3(d_out)

        d_out = self.up_sample(d_out)
        d_out = torch.cat([d_out, skips[2]], dim=1)
        d_out = self.conv_d_2(d_out)

        d_out = self.up_sample(d_out)
        d_out = torch.cat([d_out, skips[3]], dim=1)
        d_out = self.conv_d_1(d_out)
        
        d_out = self.up_sample(d_out)
        d_out = self.conv_d_0(d_out)   
        
        # final convolution later
        out = self.conv_final(d_out)        
        
        return out
