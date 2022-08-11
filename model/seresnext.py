import torch 
from torch import nn, optim 
import torch.nn.functional as F
import sys 
from torchvision.models import resnet34, resnet50

def conv3x3(in_channel, out_channel): #not change resolusion
    return nn.Conv2d(in_channel,out_channel,
                      kernel_size=3,stride=1,padding=1,dilation=1,bias=False)

def conv1x1(in_channel, out_channel): #not change resolution
    return nn.Conv2d(in_channel,out_channel,
                      kernel_size=1,stride=1,padding=0,dilation=1,bias=False)


class cSEBlock(nn.Module):
    def __init__(self, c, feat):
        super().__init__()
        self.attention_fc = nn.Linear(feat,1, bias=False)
        self.bias         = nn.Parameter(torch.zeros((1,c,1), requires_grad=True))
        self.sigmoid      = nn.Sigmoid()
        self.dropout      = nn.Dropout2d(0.1)
        
    def forward(self,inputs):
        batch,c,h,w = inputs.size()
        x = inputs.view(batch,c,-1)
        x = self.attention_fc(x) + self.bias
        x = x.view(batch,c,1,1)
        x = self.sigmoid(x)
        x = self.dropout(x)
        return inputs * x

class sSEBlock(nn.Module):
    def __init__(self, c, h, w):
        super().__init__()
        self.attention_fc = nn.Linear(c,1, bias=False)
        self.bias         = nn.Parameter(torch.zeros((1,h,w,1), requires_grad=True))
        self.sigmoid      = nn.Sigmoid()
        
    def forward(self,inputs):
        batch,c,h,w = inputs.size()
        x = torch.transpose(inputs, 1,2) #(*,c,h,w)->(*,h,c,w)
        x = torch.transpose(x, 2,3) #(*,h,c,w)->(*,h,w,c)
        x = self.attention_fc(x) + self.bias
        x = torch.transpose(x, 2,3) #(*,h,w,1)->(*,h,1,w)
        x = torch.transpose(x, 1,2) #(*,h,1,w)->(*,1,h,w)
        x = self.sigmoid(x)
        return inputs * x
    
class scSEBlock(nn.Module):
    def __init__(self, c, h, w):
        super().__init__()
        self.cSE = cSEBlock(c,h*w)
        self.sSE = sSEBlock(c,h,w)
    
    def forward(self, inputs):
        x1 = self.cSE(inputs)
        x2 = self.sSE(inputs)
        return x1+x2


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.theta    = nn.utils.spectral_norm(conv1x1(channels, channels//8))
        self.phi      = nn.utils.spectral_norm(conv1x1(channels, channels//8))
        self.g        = nn.utils.spectral_norm(conv1x1(channels, channels//2))
        self.o        = nn.utils.spectral_norm(conv1x1(channels//2, channels))
        self.gamma    = nn.Parameter(torch.tensor(0.), requires_grad=True)
        
    def forward(self, inputs):
        batch,c,h,w = inputs.size()
        theta = self.theta(inputs) #->(*,c/8,h,w)
        phi   = F.max_pool2d(self.phi(inputs), [2,2]) #->(*,c/8,h/2,w/2)
        g     = F.max_pool2d(self.g(inputs), [2,2]) #->(*,c/2,h/2,w/2)
        
        theta = theta.view(batch, self.channels//8, -1) #->(*,c/8,h*w)
        phi   = phi.view(batch, self.channels//8, -1) #->(*,c/8,h*w/4)
        g     = g.view(batch, self.channels//2, -1) #->(*,c/2,h*w/4)
        
        beta = F.softmax(torch.bmm(theta.transpose(1,2), phi), -1) #->(*,h*w,h*w/4)
        o    = self.o(torch.bmm(g, beta.transpose(1,2)).view(batch,self.channels//2,h,w)) #->(*,c,h,w)
        return self.gamma*o + inputs
    
    

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
            conv1x1(in_channel, in_channel//reduction),
            nn.ReLU(True),
            conv1x1(in_channel//reduction, in_channel)
        )
        
    def forward(self, inputs):
        x1 = self.global_maxpool(inputs)
        x2 = self.global_avgpool(inputs)
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x  = torch.sigmoid(x1 + x2)
        return x
    
    
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3x3 = conv3x3(2,1)
        
    def forward(self, inputs):
        x1,_ = torch.max(inputs, dim=1, keepdim=True)
        x2 = torch.mean(inputs, dim=1, keepdim=True)
        x  = torch.cat([x1,x2], dim=1)
        x  = self.conv3x3(x)
        x  = torch.sigmoid(x)
        return x
    
    
class CBAM(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(in_channel, reduction)
        self.spatial_attention = SpatialAttentionModule()
        
    def forward(self, inputs):
        x = inputs * self.channel_attention(inputs)
        x = x * self.spatial_attention(x)
        return x
    
    
class CenterBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = conv3x3(in_channel, out_channel)
        
    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class DecodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upsample):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module('upsample',nn.Upsample(scale_factor=2, mode='nearest'))
        self.conv3x3_1 = conv3x3(in_channel, in_channel)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.conv3x3_2 = conv3x3(in_channel, out_channel)
        self.cbam = CBAM(out_channel, reduction=16)
        self.conv1x1   = conv1x1(in_channel, out_channel)
        
    def forward(self, inputs):
        x  = F.relu(self.bn1(inputs))
        x  = self.upsample(x)
        x  = self.conv3x3_1(x)
        x  = self.conv3x3_2(F.relu(self.bn2(x)))
        x  = self.cbam(x)
        x += self.conv1x1(self.upsample(inputs)) #shortcut
        return x

#U-Net ResNet34 + CBAM + hypercolumns + deepsupervision
class UNET_RESNET34(nn.Module):
    def __init__(self, resolution, load_weights=True):
        super().__init__()
        
        #encoder
        model_name = 'resnet34' #26M
        _resnet34 = resnet34(num_classes=1000, pretrained=True)
 
        self.conv1   = _resnet34.conv1 #(*,3,h,w)->(*,64,h/2,w/2)
        self.bn1     = _resnet34.bn1
        self.maxpool = _resnet34.maxpool #->(*,64,h/4,w/4)
        self.layer1  = _resnet34.layer1 #->(*,64,h/4,w/4) 
        self.layer2  = _resnet34.layer2 #->(*,128,h/8,w/8) 
        self.layer3  = _resnet34.layer3 #->(*,256,h/16,w/16) 
        self.layer4  = _resnet34.layer4 #->(*,512,h/32,w/32) 
        
        #center
        self.center  = CenterBlock(512,512) #->(*,512,h/32,w/32) 
        
        #decoder
        self.decoder4 = DecodeBlock(512+512,64, upsample=True) #->(*,64,h/16,w/16) 
        self.decoder3 = DecodeBlock(64+256,64, upsample=True) #->(*,64,h/8,w/8) 
        self.decoder2 = DecodeBlock(64+128,64,  upsample=True) #->(*,64,h/4,w/4) 
        self.decoder1 = DecodeBlock(64+64,64,   upsample=True) #->(*,64,h/2,w/2)
        self.decoder0 = DecodeBlock(64,64, upsample=True) #->(*,64,h,w) 
        
        #upsample
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        #final conv
        self.final_conv = nn.Sequential(
            conv3x3(320,64),
            nn.ELU(True),
            conv1x1(64,1)
        )
        
    def forward(self, inputs):
        #encoder
        x0 = F.relu(self.bn1(self.conv1(inputs))) #->(*,64,h/2,w/2) 
        x0 = self.maxpool(x0) #->(*,64,h/4,w/4)
        x1 = self.layer1(x0) #->(*,64,h/4,w/4)
        x2 = self.layer2(x1) #->(*,128,h/8,w/8)
        x3 = self.layer3(x2) #->(*,256,h/16,w/16)
        x4 = self.layer4(x3) #->(*,512,h/32,w/32)
    
        #center
        y5 = self.center(x4) #->(*,512,h/32,w/32)
        
        #decoder
        y4 = self.decoder4(torch.cat([x4,y5], dim=1)) #->(*,64,h/16,w/16)
        y3 = self.decoder3(torch.cat([x3,y4], dim=1)) #->(*,64,h/8,w/8)
        y2 = self.decoder2(torch.cat([x2,y3], dim=1)) #->(*,64,h/4,w/4)
        y1 = self.decoder1(torch.cat([x1,y2], dim=1)) #->(*,64,h/2,w/2)
        y0 = self.decoder0(y1) #->(*,64,h,w)
        
        #hypercolumns
        y4 = self.upsample4(y4) #->(*,64,h,w)
        y3 = self.upsample3(y3) #->(*,64,h,w)
        y2 = self.upsample2(y2) #->(*,64,h,w)
        y1 = self.upsample1(y1) #->(*,64,h,w)
        hypercol = torch.cat([y0,y1,y2,y3,y4], dim=1)
        
        #final conv
        logits = self.final_conv(hypercol) #->(*,1,h,w)
        
        return logits