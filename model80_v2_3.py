import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Net(nn.Module):
    def __init__(self, num_classes=625, is_train=False):
        super(Net,self).__init__()
        # 3 80 80
        self.is_train = is_train
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3,2,padding=1),
            # # 64 40 40
            # nn.Conv2d(64,128,3,stride=1,padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(3,2,padding=1),
        )
        # 32 40 40
        self.layer1 = make_layers(32,64,2,False)
        # 32 40 40  
        self.layer2 = make_layers(64,128,2,True)
        # 64 20 20 
        self.layer3 = make_layers(128,256,2,True)
        # 128 10 10
        self.layer4 = make_layers(256,256,2,True)
        # 256 5 5
        self.avgpool = nn.AvgPool2d(5,padding=0)
        # 256 1 1 
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)

        features = x.div(x.norm(p=2,dim=1,keepdim=True))
        if self.is_train:
            classes = self.classifier(x)
            return features, classes
        else:
            return features


if __name__ == '__main__':
    net = Net(reid=True)
    x = torch.randn(4,3,128,64)
    y = net(x)
    import ipdb; ipdb.set_trace()