import torch
from torch import nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                              kernel_size,stride, padding)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.l_relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                              kernel_size, stride,padding)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.l_relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(out_channels, out_channels,
                              kernel_size,stride, padding)
        self.batchnorm3 = nn.BatchNorm2d(out_channels)
        self.l_relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.conv4 = nn.Conv2d(out_channels, out_channels,
                              kernel_size, stride,padding)
        self.batchnorm4 = nn.BatchNorm2d(out_channels)
        self.l_relu4 = nn.LeakyReLU(negative_slope=0.2)
        #self.pool     = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.l_relu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.l_relu2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.l_relu3(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.l_relu4(x)
        #outputs = self.pool(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        filter_ = 32
        self.filter_ = filter_
        self.half = int(filter_/2)
        self.conv1 = Conv2d(in_channels = 1,out_channels = filter_,kernel_size = 3,stride = 1,padding = 1)
        self.pool1 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv2 = Conv2d(in_channels = filter_,out_channels = filter_ * 2,kernel_size = 3,stride = 1,padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv3 = Conv2d(in_channels = filter_ * 2,out_channels = filter_ * 4,kernel_size = 3,stride = 1,padding = 1)
        self.pool3 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv4 = Conv2d(in_channels = filter_ * 4,out_channels = filter_ * 8,kernel_size = 3,stride = 1,padding = 1)
        self.pool4 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv5 = Conv2d(in_channels = filter_ * 8,out_channels = filter_ * 16,kernel_size = 3,stride = 1,padding = 1)
        
        self.t_conv1 = nn.ConvTranspose2d(in_channels = filter_ * 16,out_channels = filter_ * 8,kernel_size = 2,stride = 2)
        self.conv6 = Conv2d(in_channels = filter_ * 16,out_channels = filter_ * 8,kernel_size = 3,stride = 1,padding = 1)
        self.t_conv2 = nn.ConvTranspose2d(in_channels = filter_ * 8,out_channels = filter_ * 4,kernel_size = 2,stride = 2)
        self.conv7 = Conv2d(in_channels = filter_ * 8,out_channels = filter_ * 4,kernel_size = 3,stride = 1,padding = 1)
        self.t_conv3 = nn.ConvTranspose2d(in_channels = filter_ * 4,out_channels = filter_ * 2,kernel_size = 2,stride = 2)
        self.conv8 = Conv2d(in_channels = filter_ * 4,out_channels = filter_ * 2,kernel_size = 3,stride = 1,padding = 1)
        self.t_conv4 = nn.ConvTranspose2d(in_channels = filter_ * 2,out_channels = filter_,kernel_size = 2,stride = 2)
        self.conv9 = Conv2d(in_channels = filter_ * 2,out_channels = filter_,kernel_size = 3,stride = 1,padding = 1)
        self.conv10 = nn.Conv2d(in_channels = filter_,out_channels = 1,kernel_size = 3,stride = 1,padding = 1)
        self.fc_1 = nn.Linear(filter_*31,128)
        self.l_relu1 = nn.LeakyReLU(negative_slope = 0.2)
        # self.fc_2 = nn.Linear(128,4)
        # self.fc_4 = nn.Linear(128,2)
        self.fc_2 = nn.Linear(int(filter_//2)*31,8)
        self.fc_4 = nn.Linear(int(filter_//2)*31,2)
        self.fc_3 = nn.Linear(int(filter_//2)*31,128)
        
    def forward(self,x):
        x1 = self.conv1(x)
        x = self.pool1(x1)
        x2 = self.conv2(x)
        x = self.pool2(x2)
        x3 = self.conv3(x)
        x = self.pool3(x3)
        x4 = self.conv4(x)
        x = self.pool4(x4)
        x5 = self.conv5(x)

        x = self.t_conv1(x5)
        x = torch.cat([x,x4],dim = 1)
        x = self.conv6(x)
        x = self.t_conv2(x)
        x = torch.cat([x,x3],dim = 1)
        x = self.conv7(x)
        x = self.t_conv3(x)
        x = torch.cat([x,x2],dim = 1)
        x = self.conv8(x)
        x = self.t_conv4(x)
        x = torch.cat([x,x1],dim = 1)
        x = self.conv9(x)
        x = self.conv10(x)
        #x = self.l_relu1(x) 
        out_1 = F.adaptive_avg_pool2d(x1, (1, 1)).view(-1,self.filter_)
        out_2 = F.adaptive_avg_pool2d(x2, (1, 1)).view(-1,self.filter_*2)
        out_3 = F.adaptive_avg_pool2d(x3, (1, 1)).view(-1,self.filter_*4)
        out_4 = F.adaptive_avg_pool2d(x4, (1, 1)).view(-1,self.filter_*8)
        out_5 = F.adaptive_avg_pool2d(x5, (1, 1)).view(-1,self.filter_*16)
        # out_1 = F.adaptive_max_pool2d(x1, (1, 1)).view(-1,self.filter_)
        # out_2 = F.adaptive_max_pool2d(x2, (1, 1)).view(-1,self.filter_*2)
        # out_3 = F.adaptive_max_pool2d(x3, (1, 1)).view(-1,self.filter_*4)
        # out_4 = F.adaptive_max_pool2d(x4, (1, 1)).view(-1,self.filter_*8)
        # out_5 = F.adaptive_max_pool2d(x5, (1, 1)).view(-1,self.filter_*16)
        
        # out = torch.cat([out_1,out_2,out_3,out_4,out_5],dim=1)
        # out = torch.cat([out_1[:,:self.half],out_2[:,:self.half*2],out_3[:,:self.half*4],out_4[:,:self.half*8],out_5[:,:self.half*16]],dim=1)
        out_f = torch.cat([out_1[:,self.half:],out_2[:,self.half*2:],out_3[:,self.half*4:],out_4[:,self.half*8:],out_5[:,self.half*16:]],dim=1)
        out = torch.cat([out_1,out_2,out_3,out_4,out_5],dim=1)
        # out_f = torch.cat([out_1,out_2,out_3,out_4,out_5],dim=1)
        # out = self.fc_3(out_f)
        # out_1 = self.l_relu1(out)
        # out_2 = self.fc_2(out_1)
        # out_3 = self.fc_4(out_1)
        out_2 = self.fc_2(out_f)
        out_3 = self.fc_4(out_f)

        # out_1 = self.fc_1(out)
        # out_1 = self.l_relu1(out_1)
        # out_2 = self.fc_2(out_1)
        # out_mix = torch.cat([out,out_1],dim=1)

        # out_h = torch.cat([out_1[:,:self.half],out_2[:,:self.half*2],out_3[:,:self.half*4],out_4[:,:self.half*8],out_5[:,:self.half*16]],dim=1)
        # out_f = torch.cat([out_1[:,self.half:],out_2[:,self.half*2:],out_3[:,self.half*4:],out_4[:,self.half*8:],out_5[:,self.half*16:]],dim=1)
        # return x,out,out_h,out_f
        # return x,out,out_2,out_mix
        return x,out_2,out_3,out_f,out
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        filter_ = 32
        self.filter_ = filter_
        self.half = int(filter_/2)
        self.conv1 = Conv2d(in_channels = 1,out_channels = filter_,kernel_size = 3,stride = 1,padding = 1)
        self.pool1 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv2 = Conv2d(in_channels = filter_,out_channels = filter_ * 2,kernel_size = 3,stride = 1,padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv3 = Conv2d(in_channels = filter_ * 2,out_channels = filter_ * 4,kernel_size = 3,stride = 1,padding = 1)
        self.pool3 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv4 = Conv2d(in_channels = filter_ * 4,out_channels = filter_ * 8,kernel_size = 3,stride = 1,padding = 1)
        self.pool4 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.conv5 = Conv2d(in_channels = filter_ * 8,out_channels = filter_ * 16,kernel_size = 3,stride = 1,padding = 1)
        
        self.t_conv1 = nn.ConvTranspose2d(in_channels = filter_ * 16,out_channels = filter_ * 8,kernel_size = 2,stride = 2)
        self.conv6 = Conv2d(in_channels = filter_ * 8,out_channels = filter_ * 8,kernel_size = 3,stride = 1,padding = 1)
        self.t_conv2 = nn.ConvTranspose2d(in_channels = filter_ * 8,out_channels = filter_ * 4,kernel_size = 2,stride = 2)
        self.conv7 = Conv2d(in_channels = filter_ * 4,out_channels = filter_ * 4,kernel_size = 3,stride = 1,padding = 1)
        self.t_conv3 = nn.ConvTranspose2d(in_channels = filter_ * 4,out_channels = filter_ * 2,kernel_size = 2,stride = 2)
        self.conv8 = Conv2d(in_channels = filter_ * 2,out_channels = filter_ * 2,kernel_size = 3,stride = 1,padding = 1)
        self.t_conv4 = nn.ConvTranspose2d(in_channels = filter_ * 2,out_channels = filter_,kernel_size = 2,stride = 2)
        self.conv9 = Conv2d(in_channels = filter_,out_channels = filter_,kernel_size = 3,stride = 1,padding = 1)
        self.conv10 = nn.Conv2d(in_channels = filter_,out_channels = 1,kernel_size = 3,stride = 1,padding = 1)
        self.fc_1 = nn.Linear(filter_*31,128)
        self.l_relu1 = nn.LeakyReLU(negative_slope = 0.2)
        # self.fc_2 = nn.Linear(128,4)
        # self.fc_4 = nn.Linear(128,2)
        self.fc_2 = nn.Linear(int(filter_//2)*31,8)
        self.fc_4 = nn.Linear(int(filter_//2)*31,2)
        self.fc_3 = nn.Linear(int(filter_//2)*31,128)
        
    def forward(self,x):
        x1 = self.conv1(x)
        x = self.pool1(x1)
        x2 = self.conv2(x)
        x = self.pool2(x2)
        x3 = self.conv3(x)
        x = self.pool3(x3)
        x4 = self.conv4(x)
        x = self.pool4(x4)
        x5 = self.conv5(x)

        x = self.t_conv1(x5)
        # x = torch.cat([x,x4],dim = 1)
        x = self.conv6(x)
        x = self.t_conv2(x)
        # x = torch.cat([x,x3],dim = 1)
        x = self.conv7(x)
        x = self.t_conv3(x)
        # x = torch.cat([x,x2],dim = 1)
        x = self.conv8(x)
        x = self.t_conv4(x)
        # x = torch.cat([x,x1],dim = 1)
        x = self.conv9(x)
        x = self.conv10(x)
        #x = self.l_relu1(x) 
        out_1 = F.adaptive_avg_pool2d(x1, (1, 1)).view(-1,self.filter_)
        out_2 = F.adaptive_avg_pool2d(x2, (1, 1)).view(-1,self.filter_*2)
        out_3 = F.adaptive_avg_pool2d(x3, (1, 1)).view(-1,self.filter_*4)
        out_4 = F.adaptive_avg_pool2d(x4, (1, 1)).view(-1,self.filter_*8)
        out_5 = F.adaptive_avg_pool2d(x5, (1, 1)).view(-1,self.filter_*16)
        # out_1 = F.adaptive_max_pool2d(x1, (1, 1)).view(-1,self.filter_)
        # out_2 = F.adaptive_max_pool2d(x2, (1, 1)).view(-1,self.filter_*2)
        # out_3 = F.adaptive_max_pool2d(x3, (1, 1)).view(-1,self.filter_*4)
        # out_4 = F.adaptive_max_pool2d(x4, (1, 1)).view(-1,self.filter_*8)
        # out_5 = F.adaptive_max_pool2d(x5, (1, 1)).view(-1,self.filter_*16)
        
        # out = torch.cat([out_1,out_2,out_3,out_4,out_5],dim=1)
        # out = torch.cat([out_1[:,:self.half],out_2[:,:self.half*2],out_3[:,:self.half*4],out_4[:,:self.half*8],out_5[:,:self.half*16]],dim=1)
        out_f = torch.cat([out_1[:,self.half:],out_2[:,self.half*2:],out_3[:,self.half*4:],out_4[:,self.half*8:],out_5[:,self.half*16:]],dim=1)
        out = torch.cat([out_1,out_2,out_3,out_4,out_5],dim=1)
        # out = self.fc_3(out_f)
        # out_1 = self.l_relu1(out)
        # out_2 = self.fc_2(out_1)
        # out_3 = self.fc_4(out_1)
        out_2 = self.fc_2(out_f)
        out_3 = self.fc_4(out_f)

        # out_1 = self.fc_1(out)
        # out_1 = self.l_relu1(out_1)
        # out_2 = self.fc_2(out_1)
        # out_mix = torch.cat([out,out_1],dim=1)

        # out_h = torch.cat([out_1[:,:self.half],out_2[:,:self.half*2],out_3[:,:self.half*4],out_4[:,:self.half*8],out_5[:,:self.half*16]],dim=1)
        # out_f = torch.cat([out_1[:,self.half:],out_2[:,self.half*2:],out_3[:,self.half*4:],out_4[:,self.half*8:],out_5[:,self.half*16:]],dim=1)
        # return x,out,out_h,out_f
        # return x,out,out_2,out_mix
        return x,out_2,out_3,out_f,out



# class Dual_model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.UNet_1 = UNet()
#         self.UNet_2 = UNet()
#         self.UNet_3 = UNet()
    
#     def forward(self,x):
#         y0,_ = self.UNet_1(x[0])
#         y1,_ = self.UNet_1(x[1])
#         x0 = x[0] - y0
#         x1 = x[1] - y1
#         x2_0,out_h2 = self.UNet_2(x0)
#         x2_1,out_f2 = self.UNet_2(x1)
#         x3_0,out_h2 = self.UNet_3(y0)
#         x3_1,out_f2 = self.UNet_3(y1)
#         out_2 = x3_0 + x2_1
#         out_1 = x3_1 + x2_0
#         out_2 = out_2.view(1,-1,1,64,64)
#         out_1 = out_1.view(1,-1,1,64,64)
#         x = torch.cat([out_1,out_2],dim=0)
#         out_h = [out_h2,out_f2]
#         out = [x0,x1]
        
#         return x,out_h,out

class Dual_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.UNet_1 = UNet()
        self.UNet_2 = UNet()
    
    def forward(self,x):
        y0,out_h2 = self.UNet_1(x[0])
        y1,out_f2 = self.UNet_1(x[1])
        x0 = x[0] - y0
        x1 = x[1] - y1
        x2_0,_ = self.UNet_2(x0)
        x2_1,_ = self.UNet_2(x1)
        
        out_2 = y1 + x2_0
        out_1 = y0 + x2_1
        out_2 = out_2.view(1,-1,1,64,64)
        out_1 = out_1.view(1,-1,1,64,64)
        x = torch.cat([out_1,out_2],dim=0)
        out_h = [out_h2,out_f2]
        out = [torch.cat([x0,x1],dim=0),torch.cat([x2_0,x2_1],dim=0)]
        
        return x,out_h,out

class DeepSVDD(nn.Module):
    def __init__(self,unet,size):
        super().__init__()
        self.UNet = unet
        self.fc_1     = nn.Linear(size, 64)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.l_relu1  = nn.LeakyReLU(negative_slope=0.2)
        self.fc_2     = nn.Linear(64,8)
        self.batchnorm2 = nn.BatchNorm1d(8)
        self.l_relu2  = nn.LeakyReLU(negative_slope=0.2)
        self.fc_3     = nn.Linear(8, 64)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.l_relu3  = nn.LeakyReLU(negative_slope=0.2)
        self.fc_4     = nn.Linear(64, size)
    
    def forward(self,x):
        x,out = self.UNet(x)
        x = self.fc_1(out)
        x = self.batchnorm1(x)
        x = self.l_relu1(x)
        x = self.fc_2(x)
        x = self.batchnorm2(x)
        x = self.l_relu2(x)
        x = self.fc_3(x)
        x = self.batchnorm3(x)
        x = self.l_relu3(x)
        x = self.fc_4(x)
        return x,out       