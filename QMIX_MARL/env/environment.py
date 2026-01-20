"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : MASA-QMIX
@File : environment.py
@Author : Chenasuny
@Time : 2024/3/1 22:13
"""
import csv
import math
import random

import gym
# from gym import spaces
# from gym.utils import seeding
import utils.CTM.ctm_start as cs
from MARL.common.arguments import get_common_args
# from utils.CTM.ctm_simulate import baseGraph, nodeInfo
from utils.CTM.utils import calculate_fire_risk, load_all_firebytime
# import pandas as pd

#环境类
class DynamicSignalEnv(gym.Env):
    environment_name = "dynamic signal"
    def __init__(self,args):
        self.args = args
        self.baseGraph = None
        # 下面两个参数还不知道什么意思
        self.reward_threshold = -1000
        # self.trials = 50  # 这个就类似于steps
        self.actions_record_for_agant = []

        self.fire_current = []
        # 一个全局状态，一个观测
        self.state4marl = None  # 维护全局state的变量
        self.obs4marl = None # 维护每个智能体局部观测的变量
        self.static_field = None
        self.fire_levels = None
        self.congestion_levels = None
        self.step_cell_people_count = []
        self.reset(load_all_firebytime())


    def reset(self,fire_info=None):
        """
        初始化环境（重置环境）
        得到初始化后的全局观测和局部观测
        """
        random.seed(None)
        self.rannum = random.random()
        self.step_count = 0
        self.done = False
        # 加载火的信息

        self.all_firebytime = fire_info
        # self.fire_current = self._update_fire()
        self.baseGraph = cs.init(self.all_firebytime)
        self.episode_time_slice = []
        # 使用列表推导将二维列表转换为一维，并利用集合去除重复元素
        self.action_space = [0,1,2,3] #list(set([item for sublist in self.baseGraph.signal_available_direction for item in sublist]))

        # 从所有节点中提取出全局状态和局部可观测状态
        #获取全局状态
        self.state4marl, step0_count= self._get_state_list()
        self.step_cell_people_count.append(step0_count)
        #获取局部状态
        self.obs4marl = self._get_obs_list()

    # def _update_fire(self):
    #     """
    #     更新当前火的信息
    #     """
    #     return  [self.all_firebytime[0][self.step_count],self.all_firebytime[1][self.step_count],
    #                         self.all_firebytime[2][self.step_count],self.all_firebytime[3][self.step_count],]

    def _get_state_list(self):
        """
        全局的状态
        """
        all_nodesinfo = self.baseGraph.nodesinfo
        all_people_number = 0
        cell_people_count = []
        #静态场
        static_field = []
        fire_levels = []
        congestion_levels = []
        for i in all_nodesinfo.values():
            N = math.ceil(i.current_number)
            all_people_number = all_people_number + N
            cell_people_count.append(N)
            static_field.append(N*(round(i.energy_domine/22,3)))

            _,fire_level = calculate_fire_risk(i.current_fireinfo[0],i.current_fireinfo[1],i.current_fireinfo[2],i.current_fireinfo[3])
            fire_levels.append(fire_level*N)  #用了火灾大小

            congestion_level = 0
            if i.current_density > 2.5:  # 大于2.5为拥挤///////////
                congestion_level = round((i.current_density -2.5 )/(6-2.5),3)
            congestion_levels.append(N*congestion_level)
        self.current_people_number = all_people_number
        self.static_field = static_field
        self.fire_levels = fire_levels
        self.congestion_levels = congestion_levels
        cc = [x + y + z for x, y, z in zip(static_field ,congestion_levels, fire_levels)]
        #print("当前总人数："+str(all_people_number))
        return [all_people_number] + cc[0:242] , cell_people_count#

    def _get_obs_list(self):
        """
        每个智能体的观测
        """
        # agents_obs = []
        # all_nodesinfo = self.baseGraph.nodesinfo
        # for i in self.baseGraph.agent_cell_ids:
        #     agent_obs = []
        #     node = all_nodesinfo[i]
        #     current_fireinfo = node.current_fireinfo
        #     current_density = node.current_density
        #     agent_obs = agent_obs + current_fireinfo + [current_density]
        #     for j in range(4):
        #         if len(node.neighbor_ids[j+1])>0:    #这里只考虑一个方向只邻接一个的情况
        #             neighbor_node_oba = all_nodesinfo[node.neighbor_ids[j+1][0][0]]
        #             agent_obs = agent_obs+ neighbor_node_oba.current_fireinfo + [neighbor_node_oba.current_density]
        #         else:
        #             agent_obs = agent_obs + [0,0,0,0] + [0]
        #     agents_obs.append(agent_obs)

        all_nodesinfo = self.baseGraph.nodesinfo
        # all_people_number = 0
        # for i in all_nodesinfo.values():
        #     all_people_number = all_people_number + round(i.current_number)
        #
        # self.current_people_number = all_people_number

        fire_levels = []
        congestion_levels = []
        static_fields = []
        #cell 人数

        for i in range(len(all_nodesinfo)):
            node = all_nodesinfo[i + 1]
            _, fire_level = calculate_fire_risk(node.current_fireinfo[0], node.current_fireinfo[1],
                                                node.current_fireinfo[2], node.current_fireinfo[3])
            fire_levels.append(fire_level)  # 用了火灾风险大小
            congestion_levels.append(round(node.current_number))
            static_fields.append(node.energy_domine)
        return [static_fields[0:242] + congestion_levels[0:242] + fire_levels[0:242] for _ in self.baseGraph.agent_cell_ids] #


    def step(self, actions):

        self.step_count += 1
        self.baseGraph.from_actions_get_groupId_submatrix(actions)  # 得到所有的方向
        #启动CTM
        self.baseGraph = cs.start_Sub_CTM(self.baseGraph,self.step_count) #启动了一次CTM


        if self.current_people_number == 0  and  self.step_count > 20:

            self.done = True
            """
            with open(self.args.save_path +"/historydata/train_process/finished_actionsandstep.txt", "a") as f:
                # print("--" * 15 + "--" * 15, file=f)
                print("--" * 15 + "--" * 15, file=f)
                print(f"step: {self.step_count} 时结束", file=f)
                print("--" * 15 + "--" * 15, file=f)
            """
        # 从所有节点中提取出全局状态和局部可观测状态
        #获取全局状态
        self.state4marl,stepn_count = self._get_state_list()
        self.step_cell_people_count.append(stepn_count)
        #获取局部状态
        self.obs4marl = self._get_obs_list()

        # rewards = [0 for eve in actions]
        # print(self.current_people_number)

        #奖励计算
        # reward_fire, reward_congestion =  self._calcualate_fire_congestion_risk_reward()
        reward = -   sum(self.static_field) -  sum(self.fire_levels) - sum(self.congestion_levels)

        # if self.done:
        #     reward += 100
        # # info = {}
        # print(self.rannum)

        #不是baseline 时打开
        """
        if self.step_count == 119 and self.rannum > 0.99: #
            with open(self.args.save_path +"/historydata/train_process/rewards.txt", "a") as f:
                # if self.step_count == 1:
                print("--" * 15 + "--" * 15, file=f)
                print(f"当前步数：{self.step_count}  当前人数:{self.current_people_number} 当前奖励为：{reward} = {-sum(self.static_field)} +  {-sum(self.fire_levels)} + {-sum(self.congestion_levels)}", file=f)
                print(f"当前动作:{actions}",file=f)
                print(self.get_obs()[0],file=f)
        """


        # if self.done:
        #     reward += 300
        return  reward, self.done, [sum(self.static_field) , sum(self.fire_levels), sum(self.congestion_levels)]


    def save_env_info(self, actions):
        self.actions_record_for_agant.append(actions)


    def get_state(self):
        assert self.state4marl is not None
        return self.state4marl

    def get_obs(self):
        # assert self.obs4marl is not None and self.obs4marl != []
        return  [self.get_obs_agent(i) for i in range(len(self.baseGraph.agent_cell_ids))]


    def get_obs_agent(self, agent_id):
        return self.obs4marl[agent_id]

    def get_env_info(self):
        return {
            "n_actions": len(self.action_space ),  # 还是得把空闲动作加上去
            "n_agents": len(self.baseGraph.agent_cell_ids),  # 代理数
            "state_shape": len(self.get_state()),  # 状态数
            "obs_shape": len(self.get_obs()[0]),  #
            "episode_limit": 150  # 注意:设置太大后面全是paddings,每一回合最大的步数
        }

    def get_avail_agent_actions(self, agent_id):
        available_action = [0 for x in range(len(self.action_space))]
        for i in self.baseGraph.signal_available_direction[agent_id]:
            available_action[i-1] = 1
        return available_action



if __name__ == '__main__':
    actions = [1, 1, 2, 1, 2, 2, 0, 3, 3, 0, 0, 1, 1, 2, 3, 1, 3, 1, 0, 0, 0]
    args = get_common_args()
    fire_info = load_all_firebytime()
    env = DynamicSignalEnv(args)
    env.reset(fire_info)

    # 循环
    # rewards = []
    infos = []
    i=1
    csv_file = "output_cell_1.26.csv"
    csv_cell_count = "output_cell_count.csv"
    with open(csv_file, 'w', newline='') as file:
        while True:
            reward, done, info = env.step(actions)
            print(i)
            # rewards.append(reward)
            info.append(reward)
            print(info)
            infos.append(info)
            if done:
                print(f"finish:step{i}")
                writer = csv.writer(file)
                # 循环写入每一行数据
                for row in infos:
                    writer.writerow(row)
                print(f'CSV 文件 {csv_file} 写入完成！')

                with open(csv_cell_count, 'w', newline='') as file_cell:
                    writer2 = csv.writer(file_cell)

                    # 将二维列表中的每一行写入CSV文件
                    writer2.writerows(env.step_cell_people_count)
                break
            i+=1

