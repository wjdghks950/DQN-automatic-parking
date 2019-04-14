# -----------------------------------
# car parking agent using Q-learning
# Author: Tao Chen
# Date: 2016.11.16
# -----------------------------------

import numpy as np
import time
import random
from car_parking_env import car_sim_env
from car_parking_env import Agent
import os, sys, termios, tty
import re
from collections import namedtuple
states = namedtuple('states','x y theta_heading s theta_steering')
import cPickle
import threading

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_states, 64)
#        self.bn1 = nn.BatchNorm2d(64)
        self.fc2 = nn.Linear(64,128)
#        self.bn2 = nn.BatchNorm2d(128)
        self.fc3 = nn.Linear(128, n_actions)
#        self.bn3 = nn.BatchNorm2d(n_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        return out
        #return out.view(out.size(0), -1)


class LearningAgent(Agent):
    """An agent that learns to automatic parking"""

    def __init__(self, env, test = False):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None
        self.test = test
        if not self.test:
            self.epsilon = 0.4
        else:
            self.epsilon = 0
        #print 'epsilon:',self.epsilon

        self.learning_rate = 0.90

#        self.default_q = 0.0
        self.gamma = 0.8

        self.state = None
#        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        if torch.cuda.is_available() :
            print "using cuda..."
        else :
            print "using cpu..."
        
        self.policy_net = DQN(n_states=5, n_actions=12).to(self.device)
        self.target_net = DQN(n_states=5, n_actions=12).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        print "device", self.device
        print "q net", self.policy_net
        print "target net", self.target_net

    def optimize_model(self, state, action, reward):
        state = torch.Tensor(state, device=self.device)
#        print "action : ", action
#        print "action_int : ", car_sim_env.valid_actions.index(action)
        action = torch.LongTensor([car_sim_env.valid_actions.index(action)])
        
#        print "action_tensor : ", action
#        Q = self.policy_net(state).gather(1, action)
        Q = self.policy_net(state)
#        print "Q", Q
        Q = Q.index_select(dim=0, index=action)
#        print "Q_ind", Q
        V_next = torch.zeros(1, device=self.device)
        V_next = self.target_net(state).max()[0].detach()

        expected_Q = (V_next * self.gamma) + reward

        loss = F.smooth_l1_loss(Q, expected_Q)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


        '''
        self.Q_values = {}
        self.state = None
        self.Q_stage_one = {}
        self.Q_stage_two = {}
        self.Q_to_terminal_zero = {}
        self.Q_to_terminal_one = {}
        self.Q_to_terminal_two = {}
        self.Q_to_terminal_three = {}

        if 0 :
            self.load_q_table()
        elif 0 :
            self.init_q_table()
            ## not yet implementation

    def load_q_table(self):
        parent_path = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(parent_path, 'q_table')
        print 'restoring q tables from {}'.format(data_path)
        with open(os.path.join(data_path, 'far_region.cpickle'), 'rb') as f:
            self.Q_stage_one = cPickle.load(f)
            print '    stage one q table length:',len(self.Q_stage_one), self.Q_stage_one
        with open(os.path.join(data_path, 'near_region.cpickle'), 'rb') as f:
            self.Q_stage_two = cPickle.load(f)
            print '    stage two q table length:', len(self.Q_stage_two)
        with open(os.path.join(data_path, 'bottom_left.cpickle'), 'rb') as f:
            self.Q_to_terminal_zero = cPickle.load(f)
            print '    bottom left q table length:', len(self.Q_to_terminal_zero)
        with open(os.path.join(data_path, 'bottom_right.cpickle'), 'rb') as f:
            self.Q_to_terminal_one = cPickle.load(f)
            print '    bottom right q table length:', len(self.Q_to_terminal_one)
        with open(os.path.join(data_path, 'top_right.cpickle'), 'rb') as f:
            self.Q_to_terminal_two = cPickle.load(f)
            print '    top right q table length:', len(self.Q_to_terminal_two)
        with open(os.path.join(data_path, 'top_left.cpickle'), 'rb') as f:
            self.Q_to_terminal_three = cPickle.load(f)
            print '    top left q table length:', len(self.Q_to_terminal_three)
        print 'restoring done...'

    def init_q_table(self):
        print "There is no loading file, making Q table...".format(trial)
        init_q_value = {} ## ???????????
    
        q_table_file = os.path.join(data_path, '.cpickle')
        with open(q_table_file, 'wb') as f:
            cPickle.dump(init_q_value, f, protocol=cPickle.HIGHEST_PROTOCOL)
    
    def reset(self):
        self.state = None
        self.action = None
        self.reward = None
    '''

    def update(self):
        if self.state == None:
            agent_pose = self.env.sense()
            self.state = states(x = agent_pose[0], y = agent_pose[1], theta_heading = agent_pose[2], s = agent_pose[3], theta_steering = agent_pose[4])
        '''
        if self.env.region_idx == 2:
            print '   agent in stage one...'
            self.Q_values = self.Q_stage_one.copy()
        elif self.env.region_idx == 1:
            print '   agent in stage two...'
            self.Q_values = self.Q_stage_two.copy() ##This code resets the self.Q_values to {}
        else:
            if self.env.to_terminal_idx == 0:
                print '   agent in stage three, starting from bottom left...'
                self.Q_values = self.Q_to_terminal_zero.copy()
            elif self.env.to_terminal_idx == 1:
                print '   agent in stage three, starting from bottom right...'
                self.Q_values = self.Q_to_terminal_one.copy()
            elif self.env.to_terminal_idx == 2:
                print '   agent in stage three, starting from top right...'
                self.Q_values = self.Q_to_terminal_two.copy()
            else:
                print '   agent in stage three, starting from top left...'
                self.Q_values = self.Q_to_terminal_three.copy()

        #print 'Q_table length:',len(self.Q_values)
        #print '===================================='
        #print 'Current State: ', self.state
        '''


        step = self.env.get_steps()
        if self.env.enforce_deadline:
            deadline = self.env.get_deadline()

        # Select action according to your policy
        action = self.get_action(self.state)
        #print "action: " + str(action) + "\n"
        #time.sleep(5)

        # print 'max_q_value:',max_q_value

        # Execute action and get reward
        next_agent_pose,reward = self.env.act(self, action)
        self.next_state = states(x=next_agent_pose[0], y=next_agent_pose[1], theta_heading=next_agent_pose[2], s=next_agent_pose[3], theta_steering=next_agent_pose[4])

        # Learn policy based on state, action, reward
        if not self.test:
            self.optimize_model(self.state, action, reward)
            #self.update_q_values(self.state, action, self.next_state, reward)
            #print self.Q_values

            # if self.env.enforce_deadline:
            #     print "LearningAgent.update(): step = {}, deadline = {}, state = {}, action = {}, reward = {}".format(step, deadline,
            #                                                                                                           self.next_state, action, reward)
            # else:
            #     print "LearningAgent.update(): step = {}, state = {}, action = {}, reward = {}".format(step, self.next_state,
            #                                                                                            action, reward)
        self.state = self.next_state
    '''
    def set_q_tables(self, path):
        with open(path, 'rb') as f:
            self.Q_values = cPickle.load(f)

    '''
    def save_state(self, state, action):
    	self.prev_state = self.state
        self.prev_action = action

    '''
    def update_q_values(self, state, action, next_state, reward):
        #print "Updating Q Vale -------------"
    	old_q_value = self.Q_values.get((state,action), self.default_q)
        action, max_q_value = self.get_maximum_q_value(next_state)
    	new_q_value = old_q_value + self.learning_rate * (reward + self.gamma * max_q_value - old_q_value)
        self.Q_values[(state,action)] = new_q_value
        print "Q Lenght:  ", len(self.Q_values)
        #print "Q values++++ \n" , self.Q_values
        print "+++new Q Value+++: " + str(new_q_value)
        print "current state: ", (state,action)
    	
        # print 'Q_values.shape',len(self.Q_values)

    def get_maximum_q_value(self, state):
        q_value_selected = -10000000
        for action in car_sim_env.valid_actions:
            q_value = self.get_q_value(state, action)
            if q_value > q_value_selected:
                q_value_selected = q_value
                action_selected = action
            elif q_value == q_value_selected:  # if there are two actions that lead to same q value, we need to randomly choose one between them
                action_selected = random.choice([action_selected, action])
        return action_selected, q_value_selected
    '''
    def get_action(self, state): 
        state = torch.Tensor(state, device=self.device)
        
        if random.random() > self.epsilon :
            with torch.no_grad():
                q_values = self.policy_net(state)
                action_selected = car_sim_env.valid_actions[q_values.argmax()]
                #print "\tget_action : q_val, act", (action_selected)
        else :
            action_selected = random.choice(car_sim_env.valid_actions)

        '''
        if random.random() < self.epsilon:
            action_selected = random.choice(car_sim_env.valid_actions)
            q_value_selected = self.get_q_value(state, action_selected)
        else:
            action_selected, q_value_selected = self.get_maximum_q_value(state)
        '''
        return action_selected
    '''
    def get_q_value(self, state, action):
        #print self.Q_values
        return self.Q_values.get((state,action), self.default_q)
    '''
def run(restore):
    env = car_sim_env()
    agt = env.create_agent(LearningAgent, test=False)
    env.set_agent(agt, enforce_deadline=False)


    train_thread = threading.Thread(name="train", target=train, args=(env, agt, restore))
    train_thread.daemon = True
    train_thread.start()
    
    #train(env, agt, restore)
    env.plt_show()
    #train(env, agt, restore)
    while True:
        continue


def train(env, agt, restore):
    n_trials = 9999999999
    quit = False
    max_index = 0
    '''
    parent_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(parent_path, 'q_table')
    lfd_path = os.path.join(parent_path, 'LfD')

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    files_lst = os.listdir(data_path)
    max_index = 0
    filepath = ''
    for filename in files_lst:
        fileindex_list = re.findall(r'\d+', filename)
        if not fileindex_list:
            continue
        fileindex = int(fileindex_list[0])
        if fileindex >= max_index:
            max_index = fileindex
            filepath = os.path.join(data_path, filename)
    
    if restore:
        if os.path.exists(filepath):
            #print 'restoring Q_values from {} ...'.format(filepath)
            agt.set_q_tables(filepath)
            #print 'restoring done...'


    '''
    for trial in xrange(max_index + 1, n_trials):
        # time.sleep(3)
        #print "Simulator.run(): Trial {}".format(trial)  # [debug]
        if not agt.test:
            if trial > 80000 and trial < 150000:
                agt.epsilon = 0.3
            elif trial > 150000 and trial < 250000:
                agt.epsilon = 0.2
            elif trial > 250000:
                agt.epsilon =20000 / float(trial)  # changed to this when trial >= 2300000

        env.reset()


        while True:
            try:
                env.step()
            except KeyboardInterrupt:
                quit = True
            finally:
                if quit or env.done:
                    break


        test_interval = 100
        if trial % test_interval == 0:
            total_runs = env.succ_times + env.hit_wall_times + env.hit_car_times + env.num_hit_time_limit \
                         + env.num_out_of_time
            succ_rate = env.succ_times / float(total_runs)
            hit_cars_rate = env.hit_car_times / float(total_runs)
            hit_wall_rate = env.hit_wall_times / float(total_runs)
            hit_hard_time_limit_rate = env.num_hit_time_limit  / float(total_runs)
            out_of_time_rate = env.num_out_of_time / float(total_runs)
            print '***********************************************************************'
            print 'total runs:', total_runs
            print 'successful trials: ', env.succ_times
            print 'number of trials that hit cars', env.hit_car_times
            print 'number of trials that hit walls', env.hit_wall_times
            print 'number of trials that hit the hard time limit: ', env.num_hit_time_limit
            print 'number of trials that ran out of time: ', env.num_out_of_time
            print 'successful rate: ', succ_rate
            print 'hit cars rate: ', hit_cars_rate
            print 'hit wall rate: ', hit_wall_rate
            print 'hit hard time limit rate: ', hit_hard_time_limit_rate
            print 'out of time rate: ', out_of_time_rate
            print '***********************************************************************'
            if agt.test:
                rates_file = os.path.join(data_path, 'rates' + '.cpickle')
                rates={}
                if os.path.exists(rates_file):
                    with open(rates_file, 'rb') as f:
                        rates = cPickle.load(f)
                        os.remove(rates_file)
                rates[trial] = {'succ_rate':succ_rate, 'hit_cars_rate':hit_cars_rate, 'hit_wall_rate':hit_wall_rate, \
                                 'hit_hard_time_limit_rate':hit_hard_time_limit_rate, 'out_of_time_rate':out_of_time_rate}

                with open(rates_file, 'wb') as f:
                    cPickle.dump(rates, f, protocol=cPickle.HIGHEST_PROTOCOL)
            env.clear_count()


        if not agt.test:
            if trial % 2000 == 0:
                print "Trial {} done, saving Q table...".format(trial)
                q_table_file = os.path.join(data_path, 'trial' + str('{:010d}'.format(trial)) + '.cpickle')
                with open(q_table_file, 'wb') as f:
                    cPickle.dump(agt.Q_values, f, protocol=cPickle.HIGHEST_PROTOCOL)

            if quit:
                break


if __name__ == '__main__':
    run(restore = True)



