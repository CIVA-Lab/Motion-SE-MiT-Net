import torch
import torch.nn as nn

# double conv + bn + relu
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

# UNet network model
class UNet(nn.Module):

    def __init__(self, n_out_class):
        super().__init__()
        
        # ******************** Encoder part **********************
        self.dwn_conv1 = double_conv(3, 64)
        self.dwn_conv2 = double_conv(64, 128)
        self.dwn_conv3 = double_conv(128, 256)
        self.dwn_conv4 = double_conv(256, 512)
        self.dwn_conv5 = double_conv(512, 1024)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
 
        # ******************** Decoder part  **********************
        self.trans1 = nn.ConvTranspose2d(1024,512, kernel_size=2, stride= 2)
        self.up_conv1 = double_conv(1024,512)
        self.trans2 = nn.ConvTranspose2d(512,256, kernel_size=2, stride= 2)
        self.up_conv2 = double_conv(512,256)
        self.trans3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride= 2)
        self.up_conv3 = double_conv(256,128)
        self.trans4 = nn.ConvTranspose2d(128,64, kernel_size=2, stride= 2)
        self.up_conv4 = double_conv(128,64)
        
        #output layer
        self.out = nn.Conv2d(64, n_out_class, kernel_size=1, padding=0)
        
        
    def forward(self, input):
        
        # encoder part
        x1 = self.dwn_conv1(input)
        x2 = self.maxpool(x1)
        x3 = self.dwn_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.dwn_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.dwn_conv4(x6)
        x8 = self.maxpool(x7)
        x9 = self.dwn_conv5(x8)
  
        # decoder part
        x = self.trans1(x9)
        x = self.up_conv1(torch.cat([x,x7], 1))

        x = self.trans2(x)
        x = self.up_conv2(torch.cat([x,x5], 1))

        x = self.trans3(x)
        x = self.up_conv3(torch.cat([x,x3], 1))

        x = self.trans4(x)
        x = self.up_conv4(torch.cat([x,x1], 1))
        
        x = self.out(x)
        
        return x
