import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# load model
def get_resnet18(): 
    model = torchvision.models.resnet18()
    model.fc = nn.Linear(in_features=512, out_features=2)

    self = model

    # function to extact the multiple features
    def feature_list(x):
        out_list = []
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        y   = self.fc(out)
        return y, out_list
    
    # function to extact a specific feature
    def intermediate_forward(x, layer_index):
        #print('----------------- x.shape:',x.shape, '  layer_index:',layer_index)
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        #print('--------------- out.shape:',out.shape)
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)               
        return out

    model.feature_list = feature_list
    model.intermediate_forward = intermediate_forward

    return model

def get_densenet121():
    model = torchvision.models.densenet121()
    num_features = model.classifier.in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features,  out_features=2)])
    model.classifier = nn.Sequential(*features)
    return model

def get_vgg16():
    model = torchvision.models.vgg16()
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features,  out_features=2)])
    model.classifier = nn.Sequential(*features)
    return model

def get_vgg19():
    model = torchvision.models.vgg19()
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features,  out_features=2)])
    model.classifier = nn.Sequential(*features)
    return model

#MODELS = ['resnet18', 'densenet121', 'vgg16', 'vgg19']
def get_model(model):
    if model == 'resnet18':
        return get_resnet18()
    if model == 'densenet121':
        return get_densenet121()
    if model == 'vgg16':
        return get_vgg16()
    if model == 'vgg19':
        return get_vgg19()
    