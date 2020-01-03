import os
import time
import numpy as np
import torch
import torch.nn as nn
import csv
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from matplotlib import pyplot as plt
from collections import OrderedDict
from tqdm import tqdm


def get_file_paths(directory):
    # return a list containing paths of all files in the directory
    files_paths = []
    files = os.listdir(directory)
    files = [i for i in files]
    files.sort()

    for file in files:
        single_file_path = os.path.join(directory, file)
        files_paths.append(single_file_path)

    return files_paths


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.image = x.float()
        self.label = y.float()
    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):
        image = self.image[index]
        label = self.label[index]
        return (image, label)


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = nn.functional.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=8, block_config=(2, 2, 2, 2),
                 num_init_features=32, bn_size=4, drop_rate=0.5, num_classes=2):
        super(DenseNet, self).__init__()
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=3, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        # Linear layer
        self.fc_1 = nn.Linear(34, 2)
        # self.fc_2 = nn.Linear(16, 2)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                 m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = out.view(features.size(0), -1)
        out = self.fc_1(out)
        # out = self.fc_2(out)
        out = nn.functional.softmax(out)
        return out[:, 0: 1]


x_test_path = r"./sjtu-m3dv-medical-3d-voxel-classification/test"
paths = get_file_paths(x_test_path)
names = []
print("Loading Test Data...")
masked_test = np.empty((117, 1, 1, 40, 40, 40))
i = 0
for j in tqdm(range(582)):
    path_i = x_test_path + "\\candidate" + str(j+1) + ".npz"
    if path_i in paths:
        names.append("candidate" + str(j+1))
        raw_ct = np.load(path_i)
        voxel = raw_ct['voxel']
        seg = raw_ct['seg']
        masked_test[i, 0, 0] = (voxel * seg)[30:70, 30:70, 30:70]
        i += 1
x_test = torch.from_numpy(masked_test)

net = DenseNet()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

net.load_state_dict(torch.load('net2.pth'))
optimizer.load_state_dict(torch.load('optimizer2.pth'))
print("Previous net data loaded.")

net.eval()
test_prediction = []
for i in tqdm(range(117)):
    image = x_test[i]
    image = Variable(image)
    image = image.float()
    output = net(image)
    pred = output.view(1, -1)
    pred.float()
    test_prediction.append(pred[0].item())

with open("submission.csv", "w", newline="") as datacsv:
     csvwriter = csv.writer(datacsv, dialect = ("excel"))
     csvwriter.writerow(["Id", "Predicted"])
     for i in range(1, 117):
         csvwriter.writerow([names[i-1], test_prediction[i-1]])

print("Test data prediction updated.")

