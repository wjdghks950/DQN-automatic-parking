# -----------------------------------
# car parking simulation environment in matplotlib
# Author: Tao Chen
# Date: 2016.11.16
# -----------------------------------
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib import animation
import matplotlib.patches as mpl_patches
import math
import random
import sys, tty, termios
import threading
from matplotlib.ticker import MultipleLocator
import tools
from datetime import datetime
import time

class car_sim_env(object):
    valid_actions = ['forward', 'backward', 'left_45_forward',
            'right_45_forward', 'left_45_backward', 'right_45_backward',
            'keep', 'left', 'right'
            ]

    step_length = 0.1
#    break_pedal = 0.05
#    acc_pedal = 0.01
    speed = 0.01
    r_gear = 1 # when r_gear == -1, reverse driving
    angle = 0.785

    cur_speed = 0
    cur_velocity = step_length
    """
    valid_actions_dict = {
            valid_actions[0]: np.array([speed, 1, 0.0]), \
            valid_actions[1]: np.array([-speed, 1, 0.0]), \
            valid_actions[2]: np.array([speed, 1, angle]), \
            valid_actions[3]: np.array([speed, 1, -angle]), \
            valid_actions[4]: np.array([-speed, 1, angle]), \
            valid_actions[5]: np.array([-speed, 1, -angle]), \
            valid_actions[6]: np.array([speed, r_gear, 0.0]), \
            valid_actions[7]: np.array([-speed, r_gear, 0.0]), \
            valid_actions[8]: np.array([speed, r_gear, angle]), \
            valid_actions[9]: np.array([speed, r_gear, -angle]), \
            valid_actions[10]: np.array([-speed, r_gear, angle]), \
            valid_actions[11]: np.array([-speed, r_gear, -angle])
            }  # np.array([speed, angle])
    valid_actions_dict = {valid_actions[0]: np.array([speed, 0.0]), \
                          valid_actions[1]: np.array([-speed, 0.0]), \
                          valid_actions[2]: np.array([speed, angle]), \
                          valid_actions[3]: np.array([speed, -angle]), \
                          valid_actions[4]: np.array([-speed, angle]), \
                          valid_actions[5]: np.array([-speed, -angle])}  # np.array([speed, angle])
    """
    valid_actions_dict = {valid_actions[0]: np.array([speed, 0.0]), \
                          valid_actions[1]: np.array([-speed, 0.0]), \
                          valid_actions[2]: np.array([speed, angle]), \
                          valid_actions[3]: np.array([speed, -angle]), \
                          valid_actions[4]: np.array([-speed, angle]), \
                          valid_actions[5]: np.array([-speed, -angle]), \
                          valid_actions[6]: np.array([0, 0]), \
                          valid_actions[7]: np.array([0, angle]), \
                          valid_actions[8]: np.array([0, -angle])
                          }  # np.array([speed, angle])
    def __init__(self):
        self.done = False
        self.enforce_deadline = False

        self.rect_codes = [Path.MOVETO,
                           Path.LINETO,
                           Path.LINETO,
                           Path.LINETO,
                           Path.CLOSEPOLY]
        # self.wall_verts = np.array([[-3.53, 4.0], [4.469, 4], [4.469, -4], [-3.53, -4], [-3.53, 4.0]])
        self.car_length = 0.55
        self.car_width = 0.3889
        self.car_diagonal_length = math.sqrt(self.car_width ** 2 + self.car_length ** 2)
        self.rear_wheel_center_to_car_center = 0.2
        self.forward_radius = 0.59  # the radius of circle that the car's center goes through
        self.backward_radius = 0.42
        self.forward_turning_angle = 0.202
        self.backward_turning_angle = 0.262
        self.wall_center = np.array([0.469, 0])

        self.wall_edge_length = 8.0
        self.wall_verts = self.get_rect_verts(self.wall_center, self.wall_edge_length, self.wall_edge_length, angle=0.0)
        # self.wall_verts = self.get_rect_verts(self.wall_center, 3.6, 2.4, angle=0.0)
        self.wall_verts_closed = self.close_rect(self.wall_verts)


        self.car1_center = np.array([0.0, -2.5])
        self.car1_verts = self.get_rect_verts(self.car1_center, self.car_length, self.car_width, angle=0.0)
        self.car1_verts_closed = self.close_rect(self.car1_verts)

        self.car2_center = np.array([-0.0, -1.5])
        self.car2_verts = self.get_rect_verts(self.car2_center, self.car_length, self.car_width, angle=0.0)
        self.car2_verts_closed = self.close_rect(self.car2_verts)

        self.wall_path = Path(self.wall_verts_closed, self.rect_codes)
        self.car1_path = Path(self.car1_verts_closed, self.rect_codes)
        self.car2_path = Path(self.car2_verts_closed, self.rect_codes)

        self.env_fig = plt.figure()
        self.ax = self.env_fig.add_subplot(111, aspect='equal')
        self.wall_patch = mpl_patches.PathPatch(self.wall_path, edgecolor='black', facecolor='white', lw=5)
        self.car1_patch = mpl_patches.PathPatch(self.car1_path, facecolor='red', lw=0)
        self.car2_patch = mpl_patches.PathPatch(self.car2_path, facecolor='red', lw=0)
        self.ax.add_patch(self.wall_patch)
        self.ax.add_patch(self.car1_patch)
        self.ax.add_patch(self.car2_patch)

        self.position_noise = 0.008
        self.angle_noise = 0.008
        self.angle_blockwidth = np.pi / 8

        self.has_finished_stage_two = False
        self.region_idx = 1  # 0:close to terminal, 1:near terminal, stage two , 2:far away from terminal, stage one
        self.to_terminal_idx = 0 # 0:bottom_left  1:bottom_right   2:top_right    3:top_left

        self.get_terminal_pose()
        self.get_stage_one_terminal()
        self.get_stage_two_terminal()
        self.set_agent_start_region()

        self.done = False
        self.t = 0


        self.init_agent()
        self.ax.add_patch(self.agent_patch)
        self.ax.add_patch(self.agent_head_patch)
        self.ax.add_patch(self.agent_center_patch)

        self.succ_times = 0
        self.num_hit_time_limit = 0
        self.num_out_of_time = 0
        self.hit_wall_times = 0
        self.hit_car_times = 0
        self.hard_time_limit = 300  # even if enforce_deadline is False, end trial when deadline reaches this value (to avoid deadlocks)
        self.reward_db = []

        self.lock = threading.Lock()

    def get_terminal_pose(self):
        x_offset = 0.2
        y_offset = 0.2
        self.terminal_boundary = np.zeros(4)
        self.terminal_xy = np.zeros(2)

        self.terminal_xy[0] = 0.0
        self.terminal_xy[1] = -2.0

        self.terminal_boundary[0] = self.terminal_xy[0] - 1 * x_offset
        self.terminal_boundary[1] = self.terminal_xy[0] + 1 * x_offset
        self.terminal_boundary[2] = self.terminal_xy[1] - 1 * y_offset
        self.terminal_boundary[3] = self.terminal_xy[1] + 1 * y_offset


    def get_stage_two_terminal(self):
        x_offset = 0.049
        y_offset = 0.049
        self.stage_two_terminal_boundary = np.zeros((4, 4))
        self.stage_two_terminal_xy = np.zeros((4, 2))

        self.stage_two_terminal_xy[0][0] = 0.8
        self.stage_two_terminal_xy[0][1] = -1.4

        self.stage_two_terminal_xy[1][0] = -0.8
        self.stage_two_terminal_xy[1][1] = -1.4

        self.stage_two_terminal_xy[2][0] = -0.8
        self.stage_two_terminal_xy[2][1] = -2.6

        self.stage_two_terminal_xy[3][0] = 0.8
        self.stage_two_terminal_xy[3][1] = -2.6

        for i in range(4):
            self.stage_two_terminal_boundary[i][0] = self.stage_two_terminal_xy[i][0] - x_offset
            self.stage_two_terminal_boundary[i][1] = self.stage_two_terminal_xy[i][0] + x_offset
            self.stage_two_terminal_boundary[i][2] = self.stage_two_terminal_xy[i][1] - y_offset
            self.stage_two_terminal_boundary[i][3] = self.stage_two_terminal_xy[i][1] + y_offset


    def get_stage_one_terminal(self):

        self.stage_one_terminal_boundary = np.zeros(4)
        self.stage_one_terminal_xy = np.zeros(2)

        self.stage_one_terminal_xy[0] = 0.0
        self.stage_one_terminal_xy[1] = -2.0

        self.stage_one_terminal_boundary[0] = -1.75
        self.stage_one_terminal_boundary[1] = 1.75
        self.stage_one_terminal_boundary[2] = -3.15
        self.stage_one_terminal_boundary[3] =-0.85



    def clear_count(self):
        self.succ_times = 0
        self.hit_wall_times = 0
        self.hit_car_times = 0
        self.num_hit_time_limit = 0
        self.num_out_of_time = 0

    def step(self):
        # Update agents
        # Update agents
        self.agent.update()

        if self.done:
            return

        if self.agent is not None:
            if self.t >= self.hard_time_limit:
                print "Environment.step(): Primary agent hit hard time limit! Trial aborted."
                self.done = True
                self.num_hit_time_limit += 1

            elif self.enforce_deadline and self.t >= self.deadline:
                print "Environment.step(): Primary agent ran out of time! Trial aborted."
                self.done = True
                self.num_out_of_time += 1

            self.t += 1

    def sense(self):
        agent_pose = self.agent_pose.copy()  # [x, y, theta]
        if self.reach_stage_one_terminal(agent_pose) or self.has_finished_stage_two:
            grid_width = 0.1
            agent_pose[0] = np.floor((np.floor(agent_pose[0] / (grid_width / 2)) + 1) / 2) * grid_width
            agent_pose[1] = np.floor((np.floor(agent_pose[1] / (grid_width / 2)) + 1) / 2) * grid_width
            idx = np.floor(agent_pose[2] / (self.angle_blockwidth / 2))
            if idx % 2 == 0:
                idx = idx / 2
            else:
                idx = (idx + 1) / 2
            agent_pose[2] = idx % 16

            if self.reach_stage_two_terminal(agent_pose):
                self.has_finished_stage_two = True

            if not self.has_finished_stage_two:
                self.region_idx = 1
            else:
                self.region_idx = 0

        else:

            grid_width = 1.0
            agent_pose[0] = np.floor(agent_pose[0] / grid_width) * grid_width + 0.55 * grid_width
            agent_pose[1] = np.floor(agent_pose[1] / grid_width) * grid_width + 0.55 * grid_width
            idx = np.floor(agent_pose[2] / (self.angle_blockwidth / 2))
            if idx % 2 == 0:
                idx = idx / 2
            else:
                idx = (idx + 1) / 2
            agent_pose[2] = idx % 16

            self.region_idx = 2
        # agent_pose[2] represents the region the car's angle belongs to
        # [-11.25, 11.25) is region 0
        # [11.25, 33.75) is region 1
        # ...
        # [-33.75, -11.25) is region 15
        return agent_pose

    def act(self, agent, action):
        self.set_action(action)
        reward = 0.0

        agent_pose = self.sense()
        cur_pose = agent_pose[:2]
        prev_pose = np.array([agent.state.x, agent.state.y])

        reward -= 0.05 * self.t

        if self.collide_walls():
            print '========================================================================================'
            print 'agent hit wall'
            self.hit_wall_times += 1
            self.done = True
            reward = -20.0

        elif self.collide_fixed_cars():
            print '========================================================================================'
            print 'agent hit cars'
            self.hit_car_times += 1
            self.done = True
            reward = -5.0

        elif self.reach_terminal(agent_pose):
            reward = 40.0
            self.done = True
            self.succ_times += 1
            print '-----------------------------------------------------------------------------------------'
            print '-----------------------------------------------------------------------------------------'
            print '-----------------------------------------------------------------------------------------'
            print '-----------------------------------------------------------------------------------------'
            print "Environment.act(): Agent has reached destination!"

        elif (agent.state, action) in self.reward_db:
            reward = 3.0



        return agent_pose, reward

    def approaching_terminal(self, prev_pose, current_pose):
        cur_dist = self.cal_distance(current_pose, self.terminal_xy)
        if cur_dist > 1:
            pre_dist = self.cal_distance(prev_pose, self.terminal_xy)
            if pre_dist > cur_dist:
                return True

        return False


    def cal_distance(self, pose_one, pose_two):
        distance = np.linalg.norm(pose_one - pose_two)
        return distance



    def reach_terminal(self, pose):
        if pose[0] > self.terminal_boundary[0] and pose[0] < self.terminal_boundary[1] \
                and pose[1] > self.terminal_boundary[2]  and pose[1] < self.terminal_boundary[3] \
                and (pose[2] == 0 or pose[2] == 8):
            return True
        return False


    def reach_stage_one_terminal(self, ori_pose):
        if ori_pose[0] > self.stage_one_terminal_boundary[0] and ori_pose[0] < self.stage_one_terminal_boundary[1] \
                and ori_pose[1] > self.stage_one_terminal_boundary[2]  and ori_pose[1] < self.stage_one_terminal_boundary[3]:
            return True
        return False

    def reach_stage_two_terminal(self, pose):
        x_offset = 0.0
        y_offset = 0.0
        if pose[0] >= self.stage_two_terminal_boundary[0][0] + x_offset and pose[0] <= self.stage_two_terminal_boundary[0][1] - x_offset \
                and pose[1] > self.stage_two_terminal_boundary[0][2] + y_offset and pose[1] < self.stage_two_terminal_boundary[0][
            3] - y_offset \
                and (pose[2] == 0):
            self.to_terminal_idx = 0
            return True

        elif pose[0] >= self.stage_two_terminal_boundary[1][0] + x_offset and pose[0] <= self.stage_two_terminal_boundary[1][1] - x_offset \
                and pose[1] >= self.stage_two_terminal_boundary[1][2] + y_offset and pose[1] <= self.stage_two_terminal_boundary[1][
            3] - y_offset \
                and (pose[2] == 8):
            self.to_terminal_idx = 1
            return True

        elif pose[0] >= self.stage_two_terminal_boundary[2][0] + x_offset and pose[0] <= self.stage_two_terminal_boundary[2][1] - x_offset \
                and pose[1] >= self.stage_two_terminal_boundary[2][2] + y_offset and pose[1] <= self.stage_two_terminal_boundary[2][
            3] - y_offset \
                and (pose[2] == 8):
            self.to_terminal_idx = 2
            return True

        elif pose[0] >= self.stage_two_terminal_boundary[3][0] + x_offset and pose[0] <= self.stage_two_terminal_boundary[3][1] - x_offset \
                and pose[1] >= self.stage_two_terminal_boundary[3][2] + y_offset and pose[1] <= self.stage_two_terminal_boundary[3][
            3] - y_offset \
                and (pose[2] == 0):
            self.to_terminal_idx = 3
            return True

        return False


    def agent_step(self, cur_pose, action):
        # cur_pose:np.array([x,y,theta]) -> should be car_state:np.array([x,y,theta_heading,v,theta_steering])

        new_pose = np.zeros(3)
        theta_heading = cur_pose[2]
        theta_steering = self.valid_actions_dict[action][1]
        speed = self.valid_actions_dict[action][0]
        
        if (self.cur_speed + speed)*(self.cur_speed) < 0 : ## if pos-neg is changed
            self.r_gear = (-self.r_gear) ##
        self.cur_speed += speed ##
        self.cur_velocity = abs(self.cur_speed) ##
        
        speed_sign = self.r_gear ##

        angle_sign = 0
        if theta_steering > 0:  # turn left
            angle_sign  = 1
        elif theta_steering < 0:  # turn right
            angle_sign = -1


        delta_x = self.cur_velocity * np.cos(theta_heading)
        delta_y = self.cur_velocity * np.sin(theta_heading)
        v = self.cur_velocity
        b = self.forward_radius * 2

        new_pose[0] = cur_pose[0] + delta_x * speed_sign
        new_pose[1] = cur_pose[1] + delta_y * speed_sign
        new_pose[2] = cur_pose[2] + (v/b) * np.tan(theta_steering)
        new_pose[2] = new_pose[2] % (2 * np.pi)
        #new_pose[:2] += np.random.normal(0, self.position_noise / 4.0, 2)
        #new_pose[2] += np.random.normal(0, self.angle_noise / 4.0)

        '''
        else:
            car_radius = 0
            turning_angle = 0
            if self.cur_speed > 0:
                car_radius = self.forward_radius
                turning_angle = self.forward_turning_angle * (self.cur_velocity/self.step_length) ##
            elif self.cur_speed < 0:
                car_radius = self.backward_radius
                turning_angle = self.backward_turning_angle * (self.cur_velocity/self.step_length) ##



            rear_radius = math.sqrt(car_radius ** 2 - self.rear_wheel_center_to_car_center ** 2)


            new_pose[2] = cur_pose[2]  + angle_sign * speed_sign * turning_angle
            new_pose[2] = (new_pose[2] + 2 * np.pi) % (2 * np.pi)

            delta_x = self.rear_wheel_center_to_car_center * np.cos(theta) ## * (self.cur_velocity/self.step_length) ##
            delta_y = self.rear_wheel_center_to_car_center * np.sin(theta) ## * (self.cur_velocity/self.step_length) ##

            car_center = np.array([cur_pose[0], cur_pose[1]])
            rear_center = np.zeros(2)
            rear_center[0] = car_center[0] - delta_x
            rear_center[1] = car_center[1] - delta_y
            rear_center_to_car_center = rear_center - car_center

            tmp_angle = np.arctan2(rear_radius, self.rear_wheel_center_to_car_center)

            rotation_mtx = np.array([[np.cos(tmp_angle * (-1) * angle_sign), -np.sin(tmp_angle * (-1) * angle_sign)],
                                     [np.sin(tmp_angle * (-1) * angle_sign), np.cos(tmp_angle * (-1) * angle_sign)]])

            turing_center = car_center + np.dot(rotation_mtx, rear_center_to_car_center.T).T * \
                                         self.backward_radius / self.rear_wheel_center_to_car_center


            car_center_to_turing_center = car_center - turing_center
            rotation_mtx = np.array([[np.cos(turning_angle * angle_sign * speed_sign), -np.sin(turning_angle * angle_sign * speed_sign)],
                                     [np.sin(turning_angle * angle_sign * speed_sign), np.cos(turning_angle * angle_sign * speed_sign)]])
            new_car_center = turing_center + np.dot(rotation_mtx, car_center_to_turing_center.T).T
            new_pose[0] = new_car_center[0]
            new_pose[1] = new_car_center[1]
            new_pose[:2] += np.random.normal(0,self.position_noise,2)
            new_pose[2] += np.random.normal(0, self.angle_noise)
        '''
        self.update_agent_pose(new_pose)

    def update_agent_pose(self, pose):
        self.agent_pose = pose.copy()
        self.agent_center = self.agent_pose[:2]
        self.agent_dir = self.agent_pose[2]
        self.agent_verts = self.get_rect_verts(self.agent_center, self.car_length, self.car_width, self.agent_dir)

    def get_steps(self):
        return self.t

    def collide_fixed_cars(self):
        self.lock.acquire()
        agent_center_to_car1_center = np.linalg.norm(self.agent_center - self.car1_center)
        agent_center_to_car2_center = np.linalg.norm(self.agent_center - self.car2_center)
        car1_collision = False
        car2_collision = False
        if agent_center_to_car1_center > self.car_diagonal_length:
            # in this case, agent is not possible to collide with car1
            car1_collision = False
        else:
            car1_collision = tools.two_rects_intersect(self.agent_verts, self.car1_verts)

        if not car1_collision:
            if agent_center_to_car2_center > self.car_diagonal_length:
                car2_collision = False
            else:
                car2_collision = tools.two_rects_intersect(self.agent_verts, self.car2_verts)
        self.lock.release()
        if car1_collision or car2_collision:
            return True
        else:
            return False

    def collide_fixed_cars_with_pose(self, pose):
        agent_center = pose[:2]
        verts = self.get_rect_verts(pose[:2], self.car_length, self.car_width, pose[2])
        agent_center_to_car1_center = np.linalg.norm(agent_center - self.car1_center)
        agent_center_to_car2_center = np.linalg.norm(agent_center - self.car2_center)
        car1_collision = False
        car2_collision = False
        if agent_center_to_car1_center > self.car_diagonal_length:
            # in this case, agent is not possible to collide with car1
            car1_collision = False
        else:
            car1_collision = tools.two_rects_intersect(verts, self.car1_verts)

        if not car1_collision:
            if agent_center_to_car2_center > self.car_diagonal_length:
                car2_collision = False
            else:
                car2_collision = tools.two_rects_intersect(verts, self.car2_verts)

        if car1_collision or car2_collision:
            return True
        else:
            return False



    def collide_walls(self):
        self.lock.acquire()
        # agent_center_to_wall_center = np.linalg.norm(self.agent_center - self.wall_center)
        # if agent_center_to_wall_center < self.wall_edge_length / 2.0 - self.car_length / 2:
        #     self.lock.release()
        #     return False
        # else:
        #     wall_collision = tools.two_rects_intersect(self.agent_verts, self.wall_verts)
        #     out_of_wall = False
        #     if self.agent_center[0] > self.wall_verts[0,0] and self.agent_center[0] < self.wall_verts[1,0] \
        #         and self.agent_center[1] > self.wall_verts[2,1] and self.agent_center[1] < self.wall_verts[1,1]:
        #         out_of_wall = False
        #     else:
        #         out_of_wall = True
        #     self.lock.release()
        #     return wall_collision or out_of_wall

        wall_collision = tools.two_rects_intersect(self.agent_verts, self.wall_verts)
        out_of_wall = False
        if self.agent_center[0] > self.wall_verts[0, 0] and self.agent_center[0] < self.wall_verts[1, 0] \
                and self.agent_center[1] > self.wall_verts[2, 1] and self.agent_center[1] < self.wall_verts[1, 1]:
            out_of_wall = False
        else:
            out_of_wall = True
        self.lock.release()
        return wall_collision or out_of_wall

    def collide_walls_with_pose(self, pose):
        verts = self.get_rect_verts(pose[:2], self.car_length, self.car_width, pose[2])
        wall_collision = tools.two_rects_intersect(verts, self.wall_verts)
        out_of_wall = False
        if pose[0] > self.wall_verts[0, 0] and pose[0] < self.wall_verts[1, 0] \
                and pose[1] > self.wall_verts[2, 1] and pose[1] < self.wall_verts[1, 1]:
            out_of_wall = False
        else:
            out_of_wall = True
        return wall_collision or out_of_wall





    def get_deadline(self):
        return self.deadline





    def set_agent_start_region(self):
        self.agent_start_region = np.zeros(4)
        self.agent_start_region[0] = self.wall_verts[0,0] + self.car_length#-0.83
        self.agent_start_region[1] = self.wall_verts[1,0] - self.car_length#-0.83#0.85
        self.agent_start_region[2] = self.wall_verts[-1,1] + self.car_length#-2.6
        self.agent_start_region[3] = self.wall_verts[0,1] - self.car_length#-2.6#1.36

    def generate_agent_pose(self):
        random.seed(datetime.now())
        while True:
            x = random.uniform(self.agent_start_region[0], self.agent_start_region[1])
            y = random.uniform(self.agent_start_region[2], self.agent_start_region[3])
            # theta = np.pi
            theta = random.uniform(0, 2 * np.pi)
            if x < self.car1_verts[1,0] and x > self.car2_verts[0,0] \
                and y < self.car1_verts[0,1] and y > self.car1_verts[-1,1]:
                continue

            elif self.collide_fixed_cars_with_pose(np.array([x, y ,theta])):
                continue
            else:
                break
        return np.array([x,y,theta])



    def set_agent(self, agent, enforce_deadline=False):
        self.agent = agent
        self.enforce_deadline = enforce_deadline


    def init_agent(self):
        self.starting_pose = self.generate_agent_pose()
        self.update_agent_pose(self.starting_pose)
        delta_l = self.car_length / 2 * 3 / 5
        delta_x = delta_l * np.cos(self.agent_pose[2])
        delta_y = delta_l * np.sin(self.agent_pose[2])
        head_pose = np.zeros(2)
        head_pose[0] = self.agent_pose[0] + delta_x
        head_pose[1] = self.agent_pose[1] + delta_y
        self.agent_head_patch = plt.Circle(head_pose, 0.02, color='red')
        self.agent_center_patch = plt.Circle(self.agent_center, 0.02, color='brown')
        self.agent_patch = plt.Polygon(self.agent_verts, facecolor='cyan', edgecolor='blue')



    def update_agent_animation(self):
        self.lock.acquire()
        self.agent_patch.set_xy(self.agent_verts)

        delta_l = self.car_length / 2 * 3 / 5
        delta_x = delta_l * np.cos(self.agent_dir)
        delta_y = delta_l * np.sin(self.agent_dir)
        head_pose = np.zeros(2)
        head_pose[0] = self.agent_center[0] + delta_x
        head_pose[1] = self.agent_center[1] + delta_y
        self.agent_head_patch.center = head_pose
        self.agent_center_patch.center = self.agent_center
        self.lock.release()



    def create_agent(self, agent_class, *args, **kwargs):
        agent = agent_class(self, *args, **kwargs)
        return agent




    def get_rect_verts(self, center, length, width, angle):
        rotation_mtx = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        half_length = length / 2.0
        half_width = width / 2.0
        verts = np.array([[-half_length, half_width],   #top left
                          [half_length, half_width],    #top right
                          [half_length, -half_width],   #bottom right
                          [-half_length, -half_width]]) #bottom left
        verts_rot = np.dot(rotation_mtx, verts.T)
        verts_trans = verts_rot.T + center.reshape((1,2))
        return verts_trans.reshape((4,2))



    def close_rect(self, rect):
        return np.concatenate((rect, rect[0,:].reshape(1,2)), axis = 0)

    def animate(self, i):
        # self.agent_pose[2] = np.pi / 50 * (i % 50) * 2
        # self.agent_pose[0] = random.uniform(-3,3)
        # print '.........................................................................................'
        self.update_agent_animation()
        return [self.agent_patch, self.agent_head_patch, self.agent_center_patch]



    def plt_show(self):
        # print '...........................'
        self.anim = animation.FuncAnimation(self.env_fig, self.animate,
                                       init_func=None,
                                       frames=1000,
                                       interval=1,
                                       # repeat= False,
                                       blit=True)
        # print '...........................'
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.axis('off')
        # spacing = 1.0  # This can be your user specified spacing.
        # minorLocator = MultipleLocator(spacing)
        # # Set minor tick locations.
        # self.ax.yaxis.set_minor_locator(minorLocator)
        # self.ax.xaxis.set_minor_locator(minorLocator)
        # # Set grid to use minor tick locations.
        # self.ax.grid(which='minor')
        # plt.grid('on')

        plt.show()

    def set_action(self, action):
        time.sleep(0.1)
        cur_pose = self.agent_pose
        self.agent_step(cur_pose, action)

    def reset(self, repeat = False):
        self.done = False
        self.t = 0
        if not repeat:
            self.starting_pose = self.generate_agent_pose()
            self.update_agent_pose(self.starting_pose)
        else:
            self.update_agent_pose(self.starting_pose)

        self.deadline = self.cal_deadline(self.agent_pose[0], self.agent_pose[1])
        self.to_terminal_idx = 0
        self.region_idx = 1
        self.has_finished_stage_two = False


    def cal_deadline(self, x, y):
        dist = abs(x - self.terminal_xy[0]) / self.step_length + abs(y - self.terminal_xy[1]) / self.step_length
        deadline = max(int(dist) * 4,20)
        deadline = min(deadline, 150)
        print 'deadline:',deadline
        return deadline


    def getch(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


    def read_key(self):
        while True:
            ch = self.getch()
            if ch == '\x1b':
                ch = self.getch()
                if ch == '[':
                    ch = self.getch()
                    if ch == 'A':
                        print 'forward'
                        self.set_action('forward')
                    elif ch == 'B':
                        print 'backward'
                        self.set_action('backward')
                    elif ch == 'C':
                        print 'right_45_backward'
                        self.set_action('right_45_backward')
                    elif ch == 'D':
                        print 'left_45_backward'
                        self.set_action('left_45_backward')
            elif ch == ' ':
                print 'stop'
                self.set_action('stop')
            elif ch == 'a':
                print 'left_45_forward'
                self.set_action('left_45_forward')
            elif ch == 'd':
                print 'right_45_forward'
                self.set_action('right_45_forward')
            elif ch == 'r':
                print 'reset'
                self.reset()
            elif ch == '\x03' or ch == '\x71':  # ctrl + c or 'q'
                sys.exit()
            else:
                print ord(ch)


class Agent(object):
    """Base class for all agents."""

    def __init__(self, env):
        self.env = env
        self.state = None


    def reset(self, destination=None):
        pass

    def update(self, t):
        pass

    def get_state(self):
        return self.state


if __name__ == '__main__':
    print '============'
    car_sim = car_sim_env()
    car_sim.plt_show()






