import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# from transformers import ViTForImageClassification


# ResNet18 torchvision model
class R18Pretrained(nn.Module):
    def __init__(self):
        super(R18Pretrained, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 128)
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 12)

    def forward(self, curr_img, past_img):

        x1 = self.model(curr_img)
        x2 = self.model(past_img)
        
        out = torch.cat((x1, x2), dim=1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


class R50Pretrained(nn.Module):
    def __init__(self):
        super(R50Pretrained, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 128)
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 16)

    def forward(self, curr_img, past_img):

        x1 = self.model(curr_img)
        x2 = self.model(past_img)
        
        out = torch.cat((x1, x2), dim=1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


class InceptionV3Pretrained(nn.Module):
    def __init__(self):
        super(InceptionV3Pretrained, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 128)
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 16)

    def forward(self, curr_img, past_img):

        x1 = self.model(curr_img)
        x2 = self.model(past_img)
        
        out = torch.cat((x1[0], x2[0]), dim=1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


class ViTPretrained(nn.Module):
    def __init__(self):
        super(ViTPretrained, self).__init__()
        # self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        self.model.classifier = torch.nn.Linear(768, 128)
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 16)

    def forward(self, curr_img, past_img):

        x1 = self.model(curr_img).logits
        x2 = self.model(past_img).logits
        out = torch.cat((x1, x2), dim=1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


def get_gaussian_filter(kernel_size=3, sigma=2, channels=3):
    '''
    Gaussian 2D filters
    Code adopted from:
        Curriculum by smoothing:
            @article{sinha2020curriculum,
                title={Curriculum by smoothing},
                author={Sinha, Samarth and Garg, Animesh and Larochelle, Hugo},
                journal={arXiv preprint arXiv:2003.01367},
                year={2020}
            }
    '''
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is the product of two gaussian distributions 
    # for two different variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp( -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    if kernel_size == 3: padding = 1
    elif kernel_size == 5: padding = 2
    else: padding = 0

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels,
                                bias=False, padding=padding)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter


def get_laplaceOfGaussian_filter(kernel_size=3, sigma=2, channels=3):
    '''
    laplacian of Gaussian 2D filter
    '''
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.

    used_sigma = sigma
    # Calculate the 2-dimensional gaussian kernel which is
    log_kernel = (-1./(math.pi*(used_sigma**4))) \
                        * (1-(torch.sum((xy_grid - mean)**2., dim=-1) / (2*(used_sigma**2)))) \
                        * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*(used_sigma**2)))
       
    # Make sure sum of values in gaussian kernel equals 1.
    log_kernel = log_kernel / torch.sum(log_kernel)

    # Reshape to 2d depthwise convolutional weight
    log_kernel = log_kernel.view(1, 1, kernel_size, kernel_size)
    log_kernel = log_kernel.repeat(channels, 1, 1, 1)

    if kernel_size == 3: padding = 1
    elif kernel_size == 5: padding = 2
    else: padding = 0

    log_filter = nn.Conv2d( in_channels=channels, out_channels=channels, kernel_size=kernel_size, 
                            groups=channels, bias=False, padding=padding)

    log_filter.weight.data = log_kernel
    log_filter.weight.requires_grad = False
    
    return log_filter


class R18SimpleCBS(nn.Module):
    def __init__(self, args):
        super(R18SimpleCBS, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 128)
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 12)

        self.enable_cbs = args.use_cbs
        self.std = args.std
        self.factor = args.std_factor
        self.epoch = args.cbs_epoch
        self.kernel_size = args.kernel_size

        self.fil1 = args.fil1
        
    def forward(self, curr_img, past_img):

        if self.enable_cbs: 
            x1 = self.model(self.kernel1(curr_img))
            x2 = self.model(self.kernel1(past_img))
        else: 
            x1 = self.model(curr_img)
            x2 = self.model(past_img)
        
        out = torch.cat((x1, x2), dim=1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out

    def get_new_kernels(self, epoch_count):
        if epoch_count % self.epoch == 0 and epoch_count != 0:
            self.std *= self.factor
        if (self.fil1 == 'gau'): 
            self.kernel1 = get_gaussian_filter(kernel_size=self.kernel_size, sigma= self.std, channels=3)
        elif (self.fil1 == 'LOG'): 
            self.kernel1 = get_laplaceOfGaussian_filter(kernel_size=self.kernel_size, sigma= self.std, channels=3)

        


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.planes = planes
        self.enable_cbs = False
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_kernel = True
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def get_new_kernels(self, fil2, fil3, kernel_size, std):
        self.enable_cbs = True
        if (fil2 == 'gau'): 
            self.kernel1 = get_gaussian_filter(kernel_size=kernel_size, sigma= std, channels=self.planes)
        elif (fil2 == 'LOG'): 
            self.kernel1 = get_laplaceOfGaussian_filter(kernel_size=kernel_size, sigma= std, channels=self.planes)

        if (fil3 == 'gau'): 
            self.kernel2 = get_gaussian_filter(kernel_size=kernel_size, sigma= std, channels=self.planes)
        elif (fil3 == 'LOG'): 
            self.kernel2 = get_laplaceOfGaussian_filter(kernel_size=kernel_size, sigma= std, channels=self.planes)

    def forward(self, x):
        out = self.conv1(x)
        
        if self.enable_cbs: out = F.relu(self.bn1(self.kernel1(out)))         
        else: out = F.relu(self.bn1(out))         
        
        out = self.conv2(out)
        
        if self.enable_cbs: out = self.bn2(self.kernel2(out))
        else: out = self.bn2(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args):
               
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.std = args.std
        self.enable_cbs = args.use_cbs
        self.factor = args.std_factor
        self.epoch = args.cbs_epoch
        self.kernel_size = args.kernel_size

        self.fil1 = args.fil1
        self.fil2 = args.fil2
        self.fil3 = args.fil3

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, 128)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.enable_cbs: out = F.relu(self.bn1(self.kernel1(out)))
        else: out = F.relu(self.bn1(out))
            
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


    def get_new_kernels(self, epoch_count):
        if epoch_count % self.epoch == 0 and epoch_count != 0:
            self.std *= self.factor
        if (self.fil1 == 'gau'): 
            self.kernel1 = get_gaussian_filter(kernel_size=self.kernel_size, sigma= self.std, channels=64)
        elif (self.fil1 == 'LOG'): 
            self.kernel1 = get_laplaceOfGaussian_filter(kernel_size=self.kernel_size, sigma= self.std, channels=64)

        for child in self.layer1.children():
            child.get_new_kernels(self.fil2, self.fil3, self.kernel_size, self.std)

        for child in self.layer2.children():
            child.get_new_kernels(self.fil2, self.fil3, self.kernel_size, self.std)

        for child in self.layer3.children():
            child.get_new_kernels(self.fil2, self.fil3, self.kernel_size, self.std)

        for child in self.layer4.children():
            child.get_new_kernels(self.fil2, self.fil3, self.kernel_size, self.std)



class R18CBS(nn.Module):
    def __init__(self, args):
        super(R18CBS, self).__init__()
        self.model = ResNet(BasicBlock, [2,2,2,2], args)
        self.model.fc = nn.Linear(self.model.fc.in_features, 128)
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 12)

    def forward(self, curr_img, past_img):

        x1 = self.model(curr_img)
        x2 = self.model(past_img)
        
        out = torch.cat((x1, x2), dim=1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out
