# -----------------------------------
# utility functions to check if two rotated rectangles intersect
# Author: Tao Chen
# Date: 2016.10.28
# -----------------------------------
import numpy as np

import torch
#import torch.nn as nn
#import torch.nn.functional as F
import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
state = namedtuple('state', ('state_img', 'state_tuple'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0 

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, time_step=None):
        sample_t = random.sample(self.memory, batch_size) # Retrieve (idx, item) tuple from memory
#        for idx, sample in enumerate(sample_t):
#            for j in range(1, time_step+1):
#                sample_t[0][1] = torch.cat(sample_t[0][1], self.memory[idx][0][1]), dim=0)

    '''
    def sample(self, batch_size, time_step=None):
        sample_list = []
        sample_t = random.sample(list(enumerate(self.memory)), batch_size) # Retrieve (t, item) tuple from memory
        for idx, sample in enumerate(sample_t):
            # Each sample is returned in the form: <idx, (t, Transition_obj)>
            t = sample_t[idx][0]
            print('t: ', t)
            print('sample_t[idx][1][0] = (state): ', sample_t[idx][1][0])
            s = list(sample_t[idx][1]) #convert 'Transition' namedtuple into list for 'state' assignment operation

            s[0] = list(s[0]) #convert 'state' namedtuple into list for torch.cat assignment operation
            s[2] = list(s[2])

            for j in range(1, time_step+1):
                duplicate = t - j
                if duplicate < 0:
                    # if t < time_step, then fill the missing pieces with 0th state img
                    s[0][0] = torch.cat((s[0][0], self.memory[0][0][0]), dim=0)
                    s[2][0] = torch.cat((s[2][0], self.memory[0][2][0]), dim=0)
                else: # t >= time_step
                    s[0][0] = torch.cat((s[0][0], self.memory[t-j][0][0]), dim=0) #state -> state_img
                    s[2][0] = torch.cat((s[2][0], self.memory[t-j][2][0]), dim=0) #next_state -> state_img

            s[0] = tuple(s[0])
            s[2] = tuple(s[2])
            s = tuple(s)

            sample_t[idx] = (t, Transition(s[0], sample_t[idx][1][1], s[2], sample_t[idx][1][3]))
            sample_list.append(sample_t[idx][1])

        return sample_list
    '''
        
        return sample_t


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def get_line_coeffi(point1, point2):
    if point1[0] == point2[0]:
        A = 1
        B = 0
        C = -point1[0]
    else:
        x = np.array([point1[0], point2[0]])
        y = np.array([point1[1], point2[1]])
        a = np.vstack([x, np.ones(len(x))]).T
        m, c =np.linalg.lstsq(a, y, rcond=-1)[0] # y = mx + c
        A = m
        B = -1
        C = c
    return A, B, C


def two_rects_intersect(rect1_verts, rect2_verts):
    tolerance = 1e-5
    for line1_idx in range(4):
        for line2_idx in range(4):
            line1 = rect1_verts[line1_idx, :]
            line1 = np.vstack((line1, rect1_verts[(line1_idx + 1) % 4, :]))
            line2 = rect2_verts[line2_idx, :]
            line2 = np.vstack((line2, rect2_verts[(line2_idx + 1) % 4, :]))
            a1, b1, c1 = get_line_coeffi(line1[0, :], line1[1, :])
            a2, b2, c2 = get_line_coeffi(line2[0, :], line2[1, :])
            if abs(a1 * b2 - a2 * b1) < tolerance:
                continue
            else:
                A = np.array([[a1, b1], [a2, b2]])
                C = -np.array([[c1], [c2]])
                x, y = np.linalg.lstsq(A, C, rcond=-1)[0]
                # print (x - line1[0, 0]) * (x - line1[1, 0])
                # print (y - line1[0, 1]) * (y - line1[1, 1])
                # print (x - line2[0, 0]) * (x - line2[1, 0])
                # print (y - line2[0, 1]) * (y - line2[1, 1])
                if (x - line1[0, 0]) * (x - line1[1, 0]) <= tolerance and \
                   (y - line1[0, 1]) * (y - line1[1, 1]) <= tolerance and \
                   (x - line2[0, 0]) * (x - line2[1, 0]) <= tolerance and \
                   (y - line2[0, 1]) * (y - line2[1, 1]) <= tolerance:
                    return True
                else:
                    continue
    return False
