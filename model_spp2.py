import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def spatial_pyramid_pool(feature_maps,out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    pre_h, pre_w = feature_maps.size(2), feature_maps.size(3)
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(pre_h / out_pool_size[i]))
        w_wid = int(math.ceil(pre_w / out_pool_size[i]))
        h_pad = int(math.floor((h_wid*out_pool_size[i] - pre_h + 1)/2))
        w_pad = int(math.floor((w_wid*out_pool_size[i] - pre_w + 1)/2))
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(feature_maps)
        if(i == 0):
            spp = x.view(x.size(0),-1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp,x.view(x.size(0),-1)), 1)
    return spp

def spp(feature_maps, out_pool_size):
    '''
    replaced the max pooling module of spp by adaptive average pooling
    '''
    for i in range(len(out_pool_size)):
        avgp = nn.AdaptiveAvgPool2d(out_pool_size[i])
        x = avgp(feature_maps)
        if i == 0:
            spp = x.view(x.size(0), -1)
        else:
            spp = torch.cat((spp, x.view(x.size(0), -1)), -1)
    return spp

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out,is_downsample=False):
        super(BasicBlock,self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out,c_out,3,stride=1,padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y),True)

def make_layers(c_in,c_out,repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i ==0:
            blocks += [BasicBlock(c_in,c_out, is_downsample=is_downsample),]
        else:
            blocks += [BasicBlock(c_out,c_out),]
    return nn.Sequential(*blocks)

class PVRFNet_SPP(nn.Module):
    def __init__(self, num_classes=625, is_train=False):
        super(PVRFNet_SPP,self).__init__()
        # 3 64 64
        self.is_train = is_train
        self.output_num = [2,1]
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            # nn.AvgPool2d(3,2,padding=1)
            nn.MaxPool2d(3,2,padding=1),
        )
        # 32 32 32
        self.layer1 = make_layers(32,64,2,False)
        # 64 32 32
        self.layer2 = make_layers(64,128,2,True)
        # 128 16 16
        self.layer3 = make_layers(128,256,2,True)
        #spp_pool 256*(2*2 + 1*1) = 1280

        self.reduction = nn.Sequential(
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.avgpool(x)
        # x = spatial_pyramid_pool(x, self.output_num)
        x = spp(x, self.output_num)
        x = self.reduction(x)
        #x = x.view(x.size(0),-1)
        fine_feature = F.normalize(x, p=2, dim=1)
        #features = x.div(x.norm(p=2,dim=1,keepdim=True))
        if self.is_train:
            # x = F.relu(x)
            classes = self.classifier(x)
            return fine_feature, classes
        else:
            return fine_feature


if __name__ == '__main__':
    net = PVRFNet_SPP(reid=True)
    x = torch.randn(4,3,128,64)
    y = net(x)
    import ipdb; ipdb.set_trace()