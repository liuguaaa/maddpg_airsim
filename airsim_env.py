import math

import airsim
import json
import numpy as np
from collections import defaultdict

class Airsim_env:
    def __init__(self,uav_num):
        self.uav_n = uav_num
        self.client = airsim.MultirotorClient()
        self.uav_name = ["UAV" + str(i + 1) for i in range(self.uav_n)]
        self.target_pos = [10, 10, -10]
        self.origin_x, self.origin_y, self.origin_z = load_pos()
        self.load_uavs(self.origin_x,self.origin_y,self.origin_z)
        self.uav_pos = []
        self.action_space = [7] * self.uav_n
        self.observation_space = [3] * self.uav_n #x,y,z,x_val,y_val,z_val

        for i in range(self.uav_n):
            self.uav_pos.append([self.origin_x[i],self.origin_y[i],self.origin_z[i]])

    def reset(self):
        self.client.reset()
        self.load_uavs(self.origin_x,self.origin_y,self.origin_z)

        self.uav_pos.clear()

        for i in range(self.uav_n):
            self.uav_pos.append([self.origin_x[i],self.origin_y[i],self.origin_z[i]])

        return self.observation()

    def reward(self):
        dist = 0
        for uav_pos in self.uav_pos:
            dx = uav_pos[0] - self.target_pos[0]
            dy = uav_pos[1] - self.target_pos[1]
            dz = uav_pos[2] - self.target_pos[2]
            dist += math.sqrt(dx * dx + dy * dy + dz * dz)


        return -dist

    def observation(self):
        uav_obs = []
        for uav_name in self.uav_name:
            idx = int(uav_name[3]) - 1
            uav_obs.append([self.uav_pos[idx][0] / 10,self.uav_pos[idx][1] / 10,self.uav_pos[idx][2] / 10])
        return uav_obs





    def step(self,action): #next_state, reward, done, _
        print(f"action:{action}")

        dones = [False] * self.uav_n
        idx = 0
        for uav_action in action:
            offset = interpret_action(uav_action)
            self.uav_pos[idx][0] += offset[0]
            self.uav_pos[idx][1] += offset[1]
            self.uav_pos[idx][2] += offset[2]

            if self.uav_pos[idx][2] > 0:
                dones[idx] = True
                print("drop")
                break

            print(self.uav_pos[idx][0],self.uav_pos[idx][1],self.uav_pos[idx][2])
            if idx != 3:
                self.client.moveToPositionAsync(self.uav_pos[idx][0], self.uav_pos[idx][1], self.uav_pos[idx][2], 1, vehicle_name="UAV" + str(idx + 1))
            else :
                self.client.moveToPositionAsync(self.uav_pos[idx][0], self.uav_pos[idx][1], self.uav_pos[idx][2], 1,
                                                vehicle_name="UAV" + str(idx + 1)).join()
            idx += 1

        for uav in self.uav_name:
            if self.client.simGetCollisionInfo(vehicle_name=uav).has_collided == True:
                print("collision")
                dones[int(uav[3]) - 1] = True

        infos = []


        reward = self.reward()

        for done in dones:
            if done == True:
                reward -= 10000

        return self.observation(),reward,dones,infos





    def load_uavs(self,origin_x,origin_y,origin_z):
        for i in range(self.uav_n):
            uav_name = "UAV" + str(i + 1)
            self.client.enableApiControl(True,uav_name)
            self.client.armDisarm(True,uav_name)
            self.client.moveToPositionAsync(origin_x,origin_y,origin_z,True,uav_name)



def load_pos():
    with open('\mysimple426\settings.json') as file:
        origin_x = []
        origin_y = []
        origin_z = []
        data = json.load(file)

        UAVs = data["Vehicles"]
        for uav in UAVs:
            cur_uav = UAVs[uav]

            origin_x.append(cur_uav["X"])
            origin_y.append(cur_uav["Y"])
            origin_z.append(cur_uav["Z"])
        return origin_x,origin_y,origin_z

def get_UAV_pos(origin_x,origin_y,origin_z,client, vehicle_name="SimpleFlight"):
    state = client.simGetGroundTruthKinematics(vehicle_name=vehicle_name)
    x = state.position.x_val
    y = state.position.y_val
    z = state.position.z_val ############


    i = int(vehicle_name[3])
    x += origin_x[i - 1]
    y += origin_y[i - 1]
    z += origin_z[i - 1]
    pos = [x, y ,z]

    return pos

def interpret_action(uav_action):
    if uav_action[0] == 1:
        offset = [1, 0, 0]
    elif  uav_action[1] == 1:
        offset = [0, 1, 0]
    elif  uav_action[2] == 1:
        offset = [0, 0, 1]
    elif uav_action[3] == 1:
        offset = [-1, 0, 0]
    elif  uav_action[4] == 1:
        offset = [0, -1, 0]
    elif  uav_action[5] == 1:
        offset = [0, 0, -1]
    else:
        offset = [0, 0, 0]
    return offset