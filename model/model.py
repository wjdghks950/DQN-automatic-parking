import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class DQN(nn.Module):
    def __init__(self, vec_size, n_actions):
        super(DQN, self).__init__()
        # image_size = 80*60*4

        self.conv_img = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2), # 20*15
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, 1, 1), # 20*15
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 5, 5, 0), # 4*3
                nn.ReLU(inplace=True)
                )
        self.fc_img = nn.Sequential(
                nn.Linear(64*4*3, 512, bias=True),
#                nn.BatchNorm1d(num_features=512),
                nn.ReLU(inplace=True)
                )

        self.fc_v = nn.Sequential(
                nn.Linear(vec_size, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 512),
                nn.ReLU(inplace=True)
                )

        self.fc_out = nn.Sequential(
                nn.Linear(512, n_actions, bias=True)
                )

    def forward(self, x_img, x_v):
        x_img = x_img.view(-1,1,60,80)
#        print "x_img size", x_img.size()
        img = self.conv_img(x_img)
        img = img.view(img.size(0), -1)
        img = self.fc_img(img)

#        print "x_vec size", x_v.size()
        v = self.fc_v(x_v)

        out = img + v
        out = self.fc_out(out)
        return out # actions

'''
class DQN(nn.Module):
    def __init__(self, vec_size, n_actions):
        super(DQN, self).__init__()
        # image_size = 84*68*4

        self.conv_img = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0), # 20*16
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 4, 2, 0), # 9*7
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 0), # 7*5
                nn.ReLU(inplace=True)
                )
        self.fc_img = nn.Sequential(
                nn.Linear(64*7*5, 512, bias=True),
#                nn.BatchNorm1d(num_features=512),
                nn.ReLU(inplace=True)
                )

        self.fc_v = nn.Sequential(
                nn.Linear(vec_size, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 12),
                nn.ReLU(inplace=True),
#                nn.Linear(256, 256),
#                nn.ReLU(inplace=True),
#                nn.Linear(256, 512),
#                nn.BatchNorm1d(num_features=512),
#                nn.ReLU(inplace=True)
                )

        self.fc_out = nn.Sequential(
                nn.Linear(512, n_actions, bias=True)
                )

    def forward(self, x_img, x_v):
        img = self.conv_img(x_img)
        img = img.view(img.size(0), -1)
        img = self.fc_img(img)

        v = self.fc_v(x_v)

        out = img + v
        out = self.fc_out(out)
        return out # actions
'''

class DQN_atari(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        # image_size = 84*84*4

        self.conv = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 4, 2, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 0),
                nn.ReLU(inplace=True)
                )
        self.fc = nn.Sequential(
                nn.Linear(64*7*7, 512, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(512, n_actions, bias=True)
                )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

'''
class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_states, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.relu = nn.ReLU()

        
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        return out
        #return out.view(out.size(0), -1)
'''
