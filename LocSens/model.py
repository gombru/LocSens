import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy
import time

class Model_Multiple_Negatives(nn.Module):

    def __init__(self):
        super(Model_Multiple_Negatives, self).__init__()
        self.extra_net = MMNet()
        self.initialize_weights()

    def forward(self, img, tag, lat, lon):
        score = self.extra_net(img, tag, lat, lon)
        return score

    def initialize_weights(self):
        for l in self.extra_net.modules(): # Initialize only extra_net weights
            if isinstance(l, nn.Conv2d):
                n = l.kernel_size[0] * l.kernel_size[1] * l.out_channels
                l.weight.data.normal_(0, math.sqrt(2. / n))
                if l.bias is not None:
                    l.bias.data.zero_()
            elif isinstance(l, nn.Conv1d):
                n = l.kernel_size[0] * l.kernel_size[1] * l.out_channels
                l.weight.data.normal_(0, math.sqrt(2. / n))
                if l.bias is not None:
                    l.bias.data.zero_()
            elif isinstance(l, nn.BatchNorm2d):
                l.weight.data.fill_(1)
                l.bias.data.zero_()
            elif isinstance(l, nn.BatchNorm1d):
                l.weight.data.fill_(1)
                l.bias.data.zero_()
            elif isinstance(l, nn.Linear):
                l.weight.data.normal_(0, 0.01)
                l.bias.data.zero_()

class Model_Test_Retrieval(nn.Module):

    def __init__(self):
        super(Model_Test_Retrieval, self).__init__()
        self.extra_net = MMNet()

    def forward(self, img, tag, lat, lon, gpu):
        # Here tag is [100kx300], lat and lon [100kx1]
        # img [1xk], so I expand it
        img_batch = torch.zeros([len(tag), 300], dtype=torch.float32).cuda(gpu)
        # lat_batch = torch.zeros([len(tag), 1], dtype=torch.float32).cuda(gpu)
        # lon_batch = torch.zeros([len(tag), 1], dtype=torch.float32).cuda(gpu)
        img_batch[:,:] = img
        # lat_batch[:,:] = lat_batch
        # lon_batch[:,:] = lon_batch
        score = self.extra_net(img_batch, tag, lat, lon)
        return score

class Model_Test_Retrieval_ImgBatch(nn.Module):

    def __init__(self):
        super(Model_Test_Retrieval_ImgBatch, self).__init__()
        self.extra_net = MMNet()

    def forward(self, img, tag, lat, lon, gpu):
        # Here img is [500kx300], tag 300, lat, lon 1
        # img [1xk], so I expand it
        tag_batch = torch.zeros([len(img), 300], dtype=torch.float32).cuda(gpu)
        lat_batch = torch.zeros([len(img), 1], dtype=torch.float32).cuda(gpu)
        lon_batch = torch.zeros([len(img), 1], dtype=torch.float32).cuda(gpu)
        tag_batch[:,:] = tag
        lat_batch[:,:] = lat
        lon_batch[:,:] = lon
        score = self.extra_net(img, tag_batch, lat_batch, lon_batch)
        return score


class Model_Test_Tagging(nn.Module):

    def __init__(self):
        super(Model_Test_Tagging, self).__init__()
        self.extra_net = MMNet()

    def forward(self, img, tag, lat, lon, gpu):
        # Here tag is [100kx300]
        # Others are [1xk], so I expand them
        img_batch = torch.zeros([len(tag), 300], dtype=torch.float32).cuda(gpu)
        lat_batch = torch.zeros([len(tag), 1], dtype=torch.float32).cuda(gpu)
        lon_batch = torch.zeros([len(tag), 1], dtype=torch.float32).cuda(gpu)
        img_batch[:,:] = img
        lat_batch[:,:] = lat
        lon_batch[:,:] = lon
        score = self.extra_net(img_batch, tag, lat_batch, lon_batch)
        return score


class MMNet(nn.Module):

    def __init__(self):
        super(MMNet, self).__init__()

        self.fc_i = BasicFC(300,300)
        self.fc_t = BasicFC(300,300)
        self.fc_loc = BasicFC(2,300)

        self.fc1 = BasicFC_GN(900, 2048)
        self.fc2 = BasicFC_GN(2048, 2048)
        self.fc3 = BasicFC_GN(2048, 2048)
        self.fc4 = BasicFC_GN(2048, 1024)
        self.fc5 = BasicFC_GN(1024, 512)
        self.fc6 = nn.Linear(512, 1)


    def forward(self, img, tag, lat, lon):

        # A layer without norm for each modality
        img = self.fc_i(img)
        tag = self.fc_t(tag)
        loc = torch.cat((lat, lon), dim=1)
        loc = self.fc_loc(loc)

        # L2 normalize
        img_norm = img.norm(p=2, dim=1, keepdim=True)
        img = img.div(img_norm)
        img[img != img] = 0  # avoid nans

        tag_norm = tag.norm(p=2, dim=1, keepdim=True)
        tag = tag.div(tag_norm)
        tag[tag != tag] = 0  # avoid nans

        loc_norm = loc.norm(p=2, dim=1, keepdim=True)
        loc = loc.div(loc_norm)
        loc[loc != loc] = 0  # avoid nans

        # Set loc to 0!
        # loc = loc * 0

        # Concatenate
        x = torch.cat((img, tag), dim=1)
        x = torch.cat((loc, x), dim=1)

        # MLPm with GN
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)

        return x

class MMNet_Progressive_v2(nn.Module):

    def __init__(self):
        super(MMNet_Progressive_v2, self).__init__()

        self.fc_i = BasicFC(300,300)
        self.fc_t = BasicFC(300,300)
        self.fc_loc = BasicFC(2,300)

        self.fc1 = BasicFC_GN(900, 2048)
        self.fc2 = BasicFC_GN(2048, 2048)
        self.fc3 = BasicFC_GN(2048, 2048)
        self.fc4 = BasicFC_GN(2048, 1024)
        self.fc5 = BasicFC_GN(1024, 512)
        self.fc6 = nn.Linear(512, 1)


    def forward(self, img, tag, lat, lon, alpha):

        # A layer without norm for each modality
        img = self.fc_i(img)
        tag = self.fc_t(tag)
        loc = torch.cat((lat, lon), dim=1)
        loc = self.fc_loc(loc)

        # L2 normalize
        img_norm = img.norm(p=2, dim=1, keepdim=True)
        img = img.div(img_norm)
        img[img != img] = 0  # avoid nans

        tag_norm = tag.norm(p=2, dim=1, keepdim=True)
        tag = tag.div(tag_norm)
        tag[tag != tag] = 0  # avoid nans

        loc_norm = loc.norm(p=2, dim=1, keepdim=True)
        loc = loc.div(loc_norm)
        loc[loc != loc] = 0  # avoid nans

        # Progressive location(alpha [0-1])
        loc = loc * alpha

        # Concatenate
        x = torch.cat((img, tag), dim=1)
        x = torch.cat((loc, x), dim=1)

        # MLPm with GN
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)

        return x



class BasicFC(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicFC, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.fc(x)
        return F.relu(x, inplace=True)

class BasicFC_BN(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicFC_BN, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001) # momentum = 0.0001

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class BasicFC_GN(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicFC_GN, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.gn = nn.GroupNorm(32, out_channels, eps=0.001)  # 32 is num groups

    def forward(self, x):
        x = self.fc(x)
        x = self.gn(x)
        return F.relu(x, inplace=True)


class BasicFC_GN_LK(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicFC_GN_LK, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.gn = nn.GroupNorm(25, out_channels, eps=0.001)  # 32 is num groups

    def forward(self, x):
        x = self.fc(x)
        x = self.gn(x)
        return F.relu(x, inplace=True)