"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : MASA-QMIX
@File : utils.py
@Author : Chenasuny
@Time : 2024/4/2 18:16
"""
import copy
import csv
import math
import multiprocessing
import os
import random
import warnings
from pathlib import Path

# import threading
import numpy as np
from MARL.agent.agent import Agents
from MARL.common.rollout import RolloutWorker


# free_velocity = 1.2  # 自由流速度
# l = 1.7
# max_density = 6  # 最大密度值 单位：人/米方
# cri_density = 2.5  # 临界密度值 单位：人/米方
# duration = l / free_velocity  # 时间步长，单位：秒
#
#
#
# def ped_fd(density: float):  # 跟据线性的行人流基本图，输入密度，输出流率
#     if density <= cri_density:
#         return density * free_velocity
#     elif density <= max_density:
#         return (10 - density) / (10 - cri_density) * cri_density * free_velocity
#     else:
#         return (10 - max_density) / (10 - cri_density) * cri_density * free_velocity

# cri_density = 0.54
# max_density = 1.9
max_density = 6  # 最大密度值 单位：人/米方
cri_density = 2.5  # 临界密度值 单位：人/米方
free_velocity = 1.25  # 自由流速度
l = 4
# max_density = 6  # 最大密度值 单位：人/米方
# cri_density = 1.75  # 临界密度值 单位：人/米方
duration = l / free_velocity  # 时间步长，单位：秒
max_density_luggage = 2.75  # 带行李箱的行人最大密度值 单位：人/米方 https://doi.org/10.1016/j.firesaf.2007.12.005
conf = max_density / max_density_luggage





# def ped_fd(density: float):  # 行人流基本图，输入密度，输出速度  Emergency Movement: SFPE handbook of fire protection engineering, fifth edition.
#     if density <= cri_density:
#         return 1.4-0.3724 * cri_density
#     elif density <= max_density:
#         return 1.4-0.3724 * density
#     else:
#         return 1.4-0.3724 * max_density
#
# def ped_str_fd(density: float):  # 楼梯上的上行的行人基本图，输入密度，输出速度 Emergency Movement: SFPE handbook of fire protection engineering, fifth edition.
#     if density <= cri_density:
#         return 1.23-0.32718 * cri_density
#     elif density <= max_density:
#         return 1.23-0.32718 * density
#     else:
#         return 1.23-0.32718 * max_density

def ped_fd(density: float):  # 行人流基本图，输入密度，输出流率
    if density <= 2.5:
        return 1.2 * density
    elif density <= max_density:
        return 3 - 0.4 * (density -2.5)
    else:
        return 1.6

def ped_str_fd(density: float):  # 楼梯上的上行的行人基本图，输入密度，输出流率
    if density == 0:
        return 0
    else:
        return (-0.132 * np.log(density) + 0.69) * density













#
#
# def ped_fd(density: float):  # 行人流基本图，输入密度，输出速度
#     if density <= cri_density:
#         return free_velocity
#     elif density <= max_density:
#         return -0.397 + 2.88225 / density
#     else:
#         return 0.5 / density
#
#
# def ped_lug_fd(density: float):  # 带行李的行人基本图 https://doi.org/10.1016/j.physa.2018.09.038    Fig 6
#     if density <= 1.4:
#         return free_velocity
#     elif density <= 4.8:
#         return -0.3676 + 2.2645 / density
#     else:
#         return 0.5 / density
#
#
# def ped_str_fd(density: float):  # 楼梯上的上行的行人基本图 https://doi.org/10.1016/j.ssci.2021.105409  A3(C)
#     if density<=0:
#         return 0.667
#     elif 0<density <= 1.0655:
#         return 0.667 + 0.02 / density
#     elif density <= 2.25:
#         return -0.478 + 1.24 / density
#     else:
#         return 0.1645 / density
#
#
# def ped_str_lug_fd(density: float):  # 楼梯上带行李的上行的行人基本图 https://doi.org/10.1016/j.physa.2021.125880  Fig 10
#     if density <= 1.1:
#         return 0.554
#     elif density <= 1.75:
#         return -0.7077 + 1.3885 / density
#     elif density <= 2.75:
#         return -0.05 + 0.2375 / density
#     else:
#         return 0.1 / density




def calculate_fire_risk(Rad, Tem, Tox, Vis):
    """
    火灾对人的风险计算
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        beta1, beta2, beta3, beta4 = 1, 1, 1, 1
        r_rad = 1/(1+ np.exp(beta1 * (2.5- Rad)))
        r_tem = 1/(1+ np.exp(beta2 * (100- Tem)))
        r_tox = 1/(1+ np.exp(beta3 * (2800 - Tox)))
        r_vis = 1/(1+ np.exp(beta4 * (Vis - 5)))
        risk = round((r_rad+r_tem+r_tox+r_vis)/4,5) # using average score

    return float(risk), float(risk)   #没有用火灾等级


def fire_effect_speed( Tem, Tox, Vis, interal_time, v_current):
    """
    火灾对元胞间流动速度的影响
    interal_time:时间间隔
    v_current: 不考虑的情况下的速度

    return : 返回火对速度的影响系数
    """
    if v_current<1.2:
        v_current=1.2
    v_max = 2.0 #逃生时的最大速度

    e_vis, e_co, e_temp = 0,0,0
    if Vis>=7.5:
        e_vis = 1
    elif Vis<7.5 and Vis >=2.5:
        e_vis = 1.375-(2.8125/Vis)
    elif  Vis < 2.5:
        e_vis = 0.25

    if Tox<0.1:
        e_co = 1
    elif Tox >= 0.1 and Tox <= 0.25:
        e_co = 1- (0.2125+1.788*Tox)*Tox*interal_time
    elif  Tox >0.25:
        e_co = 0

    if Tem<=30:
        e_temp = 1
    elif Tem<=60 and Tem>30:
        e_temp = 1+ (v_max-v_current)*(((Tem-30)/(60-30))**2)/v_current
    elif Tem<=120 and Tem>60:
        e_temp = (v_max/v_current)*(1-(((Tem-60)/(120-60))**2))
    else:
        e_temp = 0
    return e_vis*e_co*e_temp


def random_people(total_rooms=266):
    room_list = []
    segment_totals = {
        'segment_1': 0,  # for rooms 1 to 99 and 104 to 176
        'segment_2': 0,  # for rooms 100 to 103
        'segment_3': 0,  # for rooms 177 to 184
        'segment_4': 0,  # for rooms 185 to 242
        'segment_5': 0 # for rooms 243 to 266
    }

    for room_number in range(1, total_rooms + 1):
        random.seed(12)
        if room_number > 242:
            people = 38
            segment_totals['segment_5'] += people
        elif 185 <= room_number <= 242:
            people = random.randint(0, 0)
            segment_totals['segment_4'] += people
        elif 177 <= room_number <= 184:
            people = random.randint(0, 0)
            segment_totals['segment_3'] += people
        elif 100 <= room_number <= 103:
            people = 0
            segment_totals['segment_2'] += people
        else:
            people = random.randint(0, 0)
            segment_totals['segment_1'] += people

        room_list.append(people)
    room_list[0:242]=[8,7,8,0,1,5,3,1,8,3,10,1,9,1,0,1,4,10,2,4,2,1,3,8,5,10,4,4,4,0,2,0,4,6,6,8,5,10,7,2,6,3,9,5,5,2,2,0,2,4,6,8,7,9,5,5,1,6,4,0,6,5,4,5,9,7,5,5,2,0,10,10,2,3,1,2,4,3,8,10,0,0,3,10,2,0,10,4,1,9,4,4,5,1,1,7,1,3,5,0,0,0,0,1,6,8,0,6,8,5,3,3,2,7,9,1,8,7,5,4,4,2,0,6,3,5,4,7,6,3,5,1,9,7,3,8,4,6,1,9,10,0,5,0,8,8,9,6,1,0,9,1,8,2,10,0,6,9,5,3,9,1,10,6,10,9,1,8,3,9,10,10,5,5,9,2,9,8,9,8,7,7,7,5,9,7,9,10,9,7,8,10,9,6,7,10,8,6,8,7,9,9,6,9,7,9,8,6,9,6,8,5,10,6,10,6,9,5,8,10,7,9,7,6,7,10,7,10,7,5,5,9,8,5,9,9,10,9,10,8,6,6]
    # total = sum(room_list)
    # # # 输出结果
    # print("人数为:", total)
    # # 定义文件名
    # filename = "numbers.txt"
    #
    # # 将列表中的数字转换为字符串，并用逗号隔开
    # numbers_str = ','.join(map(str, room_list))
    #
    # # 打开文件并写入字符串
    # with open(filename, 'w') as file:
    #     file.write(numbers_str)
    #
    # print(f"数字已成功保存到 {filename}")
    # # # Print the total number of people in each segment
    # # print(f"Total people in rooms 1-99 and 104-176: {segment_totals['segment_1']}")
    # # print(f"Total people in rooms 100-103: {segment_totals['segment_2']}")
    # # print(f"Total people in rooms 177-184: {segment_totals['segment_3']}")
    # # print(f"Total people in rooms 185-242: {segment_totals['segment_4']}")
    return room_list

def cell_type_create():
    cell_type_list = []
    for room_number in range(1, 266 + 1):
        if 164 <= room_number <= 176:
            people = 1
        else:
            people = 0
        cell_type_list.append(people)
    return cell_type_list

def load_all_firebytime(fire_csv_path=None):
    base_dir = Path(__file__).resolve().parents[3]
    if fire_csv_path is None:
        file_path = base_dir / 'fire_info' / 'subway_devc_fire_in_cell_251 6.csv'
    else:
        file_path = Path(fire_csv_path)
    # ????????
    if file_path.is_file():
        all_fireinfo = []
        with file_path.open('r') as file:
            reader = csv.reader(file)
            for step, row in enumerate(reader):
                if step < 2:  # 跳过前两行
                    continue
                step_fire = []
                row = [round(float(element), 5) for element in row]  # 将str转换为float

                rad = [max(round(element * 1000), 0) for element in row[799:1065]]
                rad[99] = rad[100] = rad[101] = rad[102] = 0
                #rad[243:267] = [0] * (267 - 243)
                step_fire.append(rad)  # radiation

                temp = row[267:533]
                temp[99] = temp[100] = temp[101] = temp[102] = 20
                #temp[243:267] = [20] * (267 - 243)
                step_fire.append(temp)  # temperature

                tox = row[533:799]
                tox[99] = tox[100] = tox[101] = tox[102] = 0
                #tox[243:267] = [0] * (267 - 243)
                step_fire.append(tox)  # toxic

                vis = row[1:267]
                vis[99] = vis[100] = vis[101] = vis[102] = 30
                #vis[243:267] = [30] * (267 - 243)
                step_fire.append(vis)  # visibility

                all_fireinfo.append(step_fire)
        return all_fireinfo
    else:
        # 文件不存在时返回的默认数据
        default_data = [
            [[0] * 266, [20] * 266, [0] * 266, [30] * 266] for _ in range(500)
        ]
        return default_data

#多线程的类
class multiTask:
    def __init__(self, n_episodes, env,args ,policy, fire):
        self.env = env
        self.args = copy.deepcopy(args)
        self.args.load_model = False

        self.n_episodes = n_episodes
        self.agents= Agents(self.args)
        self.agents.agents_copy(policy)
        # env = DynamicSignalEnv()
        # self.rolloutWorker = rolloutWorker
        # print(self.agents.policy.eval_rnn.state_dict())
        self.fire = fire

    def generate_episode(self, params):
        episode_idx, epoch = params
        rolloutWorker = RolloutWorker(self.env, self.agents, self.args)
        # 生成单个episode
        episode, _, _, for_gantt_data = rolloutWorker.generate_episode(epoch=epoch, episode_num=episode_idx,
                                                                            fire_info=self.fire)
        del rolloutWorker
        # print(episode_idx)
        return episode, sum(episode['r'][0])[0]

    def run_simulation(self, epoch):
        episodes = []
        r_s = []
        batch_size = 28  #同时用到的核数  #60
        params = [(episode_idx, epoch) for episode_idx in range(self.n_episodes)]
        # 使用批处理逐批执行
        for i in range(0, len(params), batch_size):
            with multiprocessing.Pool(batch_size) as pool:
                batch_results = pool.map(self.generate_episode, params[i:i+batch_size])
            for episode, r in batch_results:
                episodes.append(episode)
                r_s.append(r)
        return episodes, r_s

# #
# def CTM_simulate(graph):   # CTM，启动！
#     graph.performance = 0  #步长
#     while True:
#         graph.performance += 1
#         need_continue = False
#         for i in range(graph.max_hier):
#             for n in graph.nodes:
#                 if n.hierarchy == i:
#                     n.receive_inflow()
#                     n.update_density()
#                     if n.outflow > 0:
#                         n.clearance_time = graph.performance + n.outflow / (
#                                 n.boundary * free_velocity * cri_density * duration)
#                     if not need_continue and n.inflow != 0:
#                         need_continue = True
#         if not need_continue:
#             print(graph.performance)
#             break









if __name__ == '__main__':
    risk = calculate_fire_risk(3,130,3500,2)




