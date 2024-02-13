from torch import nn

import sys
sys.path.append("C:/Object-Detection/")

from PoseEstimation import data

class EncoderDecoderNet(nn.Module):
    def __init__(self):
        super(EncoderDecoderNet,self).__init__()
        #Convolution 1
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64, kernel_size=7,stride=2, padding=3)
        nn.init.normal_(self.conv1.weight, std=0.001)
        nn.init.constant_(self.conv1.bias, 0)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1= nn.ReLU()
        self.maxpool1= nn.MaxPool2d(kernel_size=2,stride=2)

        #Convolution 2
        self.conv2=nn.Conv2d(in_channels=64,out_channels=128, kernel_size=5,stride=1, padding=2)
        nn.init.normal_(self.conv2.weight, std=0.001)
        nn.init.constant_(self.conv2.bias, 0)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2= nn.ReLU()
        self.maxpool2= nn.MaxPool2d(kernel_size=2,stride=2)
        
        #Convolution 3
        self.conv3=nn.Conv2d(in_channels=128,out_channels=256, kernel_size=5,stride=1, padding=2)
        nn.init.normal_(self.conv3.weight, std=0.001)
        nn.init.constant_(self.conv3.bias, 0)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3= nn.ReLU()
        self.maxpool3= nn.MaxPool2d(kernel_size=2,stride=2)
        
        #Deconvolution 4
        self.deconv4 =nn.ConvTranspose2d(in_channels=256,out_channels=256,padding=1, output_padding=0,kernel_size=4, stride=2)
        nn.init.normal_(self.deconv4.weight, std=0.001)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4= nn.ReLU(inplace=True)
        
        #Deconvolution 5
        self.deconv5 =nn.ConvTranspose2d(in_channels=256,out_channels=256,padding=1, output_padding=0,kernel_size=4, stride=2)
        nn.init.normal_(self.deconv5.weight, std=0.001)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5= nn.ReLU(inplace=True)
        
        ### final layer declaration
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=17, kernel_size=1) # 17 joints -> out_channel=17
        nn.init.normal_(self.conv6.weight, std=0.001)
        nn.init.constant_(self.conv6.bias, 0)
        
    def forward(self,x):
            # conv layer
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.maxpool1(x)
            
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.maxpool2(x)
            
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.maxpool3(x)
            
            # deconv layer
            x = self.deconv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            
            x = self.deconv5(x)
            x = self.bn5(x)
            x = self.relu5(x)
            
            # final layer
            x = self.conv6(x)
            return(x)

model = EncoderDecoderNet()
# check input and output dimensions
sample = next(iter(data.val_loader))
img = sample['input_img'][None,0]
print("input dims: ", img.shape)
print("output dims: ", model(img).shape)