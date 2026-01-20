"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : MASA-QMIX
@File : graph_node.py
@Author : Chenasuny
@Time : 2024/4/1 22:21
"""
import math
from shapely.geometry import Polygon
from utils.CTM.utils import ped_fd, fire_effect_speed, ped_str_fd

free_velocity = 1.2  # 自由流速度
l = 4
max_density = 6  # 最大密度值 单位：人/米方
cri_density = 2.5  # 临界密度值 单位：人/米方
duration = l / free_velocity  # 时间步长，单位：秒


class Node:
    def __init__(self, _id, _polygon, _num, _ex, _real_id,_fire_info,_node_type):
        # 所有固定信息
        self.id = _id  # 节点id
        self.polygon = _polygon  # 节点所对应的元胞的几何信息，调用类 from shapely.geometry import Polygon
        self.exit = _ex  # 是否为出口处的虚拟节点，该节点无流出，面积为无限大
        if not self.exit:
            self.area = float(Polygon(_polygon).area)*1000000 # / 100  # 面积
        else:
            self.area = float("inf")
        self.initial_number = _num  # 疏散开始时元胞内行人数
        self.initial_density = float(_num) / float(self.area)  # 疏散开始时元胞内行人密度

        self.accessible = True  # 是否仍可用，受hazard信息影响
        self.neighbors = []  # 所有邻接节点列表，当前用不到
        self.boundaries = []  # 与所有邻接节点之间的边界值列表，当前用不到
        self.real_ID = _real_id #真实的节点编号，即从1开始

        # 所有非固定信息
        self.density = self.initial_density
        self.downstream = None  # 当前的下游节点
        self.boundary = None  # 与当前下游节点之间的边界
        self.upstreams = []  # 所有上游节点
        self.inflow = None  # 总流入量
        self.hierarchy = None  # 密度更新优先级，与到出口距离相关
        self.original_outflow = None  # 在不考虑下游情况下，理论可产生的行人流出量
        self.outflow = 0  # 实际产生的行人流出量
        self.clearance_time = None  # 清空时间
        self.fire_info = _fire_info #火的四个信息
        self.fire_effect_speed = self._fire_effect_speed_rate()
        self.node_type = _node_type

    # def get_downstream(self, other, bound):  # 获取下游
    #     self.downstream = other
    #     other.upstreams.append(self)
    #     self.boundary = bound

    def generate_flow(self):  # 生成前向流量
        if self.node_type == 0:
            flow_speed = ped_fd(self.density)
        else:
            flow_speed = ped_str_fd(self.density)
        self.original_outflow = min(self.boundary * flow_speed * duration * self._fire_effect_speed_rate(), self.density * self.area)
        return self.original_outflow

    def adjust_flow(self, reduce_factor):  # 前向流量调节
        self.outflow = math.ceil(self.original_outflow * reduce_factor)
        return self.outflow

    def receive_inflow(self):  # 每个元胞流入量生成
        if max_density > self.density - self.outflow / self.area:
            left_capacity = (max_density - self.density) * self.area + self.outflow
        else:
            left_capacity = 0
        total_inflow = 0
        self.inflow = 0
        reduce_fac = 1
        for up in self.upstreams:
            total_inflow += up.generate_flow()
        if total_inflow > left_capacity:
            reduce_fac = float(left_capacity) / float(total_inflow)
        for up in self.upstreams:
            self.inflow += up.adjust_flow(reduce_fac)

    def update_density(self):  # 每个元胞根据流入量和流出量更新行人密度
        if self.exit:
            self.outflow = 0
        self.density += (self.inflow - self.outflow) / self.area
        if self.density < 0:
            self.density = 0

    def _fire_effect_speed_rate(self):
        fire_info = self.fire_info
        return fire_effect_speed(fire_info[1],fire_info[2],fire_info[3], duration , ped_fd(self.density) )


class Graph:
    def __init__(self, exit_inf, cell_inf, num_inf, adjacency_mat,group_id_list,group_directions,fire_info,cell_type):
        # 初始化所有节点，需输入：是否是出口节点（bool型）的列表、几何信息的列表、行人树的列表、邻接矩阵
        self.nodes = []   #节点编号
        for n in range(len(cell_inf)):
            if exit_inf[n] == True:
                self.nodes.append(Node(_id=n, _polygon=cell_inf[n], _num=num_inf[n], _ex=True, _real_id=group_id_list[n]+1,_fire_info = fire_info[n], _node_type = cell_type[n]))
            else:
                self.nodes.append(Node(_id=n, _polygon=cell_inf[n], _num=num_inf[n], _ex=False, _real_id=group_id_list[n]+1,_fire_info =fire_info[n], _node_type = cell_type[n]))
        self.exs_list = exit_inf
        self.am = adjacency_mat #邻接矩阵
        self.max_hier = None
        self.performance = 0
        self.node_directions = group_directions #得到所有的方向对, 从0开始的
        self.group_id_list = group_id_list #group中所有的原始编号，索引对应了am中的位置，从0开始的
        # self.get_neighbor()
        self.load_directions()
        self.get_hierarchy() #得到最大的树级



    def load_hazard(self, hazard_inf):  # 为所有节点加载灾情信息，需输入可达性（bool型）列表
        for h in range(hazard_inf):
            if hazard_inf[h] == 1:
                self.nodes[h].accessible = False

    def get_neighbor(self):  # 根据邻接矩阵，为所有节点获取邻域信息，当前用不到
        for i in range(len(self.am)):
            for j in range(len(self.am[i])):
                if self.am[i][j][1] > 0.1 and self.nodes[j].accessible:
                    self.nodes[i].neighbors.append(self.nodes[j])
                    self.nodes[i].boundaries.append(self.am[i][j])

    def load_directions(self):  # 为所有节点添加上下游信息，需输入：指向信息（字典形式）{上游节点对象：下游节点对象}  ###这是对应的动作，应该设置一个可达动作矩阵
        for x,y in self.node_directions:
            node_x = None
            node_y = None
            for node in self.nodes:
                if node.real_ID == x:
                    node_x = node
                if node.real_ID == y:
                    node_y = node
            node_x.downstream = node_y
            node_x.boundary = self.am[self.group_id_list.index(x-1)][self.group_id_list.index(y-1)]
            node_y.upstreams.append(node_x)

    def get_hierarchy(self):   # 为所有节点计算密度更新优先级
        exist_exit = -1
        for n in self.nodes:
            if n.exit == True:
                n.hierarchy = 0
                exist_exit=1
            else:
                n.hierarchy = None
        if exist_exit == -1: #没有出口需要自己需要根节点
            # 获取所有起始节点和目标节点
            source_nodes = set()
            target_nodes = set()
            for edge in self.node_directions:
                source_nodes.add(edge[0])
                target_nodes.add(edge[1])
            # 寻找可能的根节点编号
            possible_root_nodes = target_nodes - source_nodes
            for n in self.nodes:
                if n.real_ID == list(possible_root_nodes)[0]:
                    n.hierarchy = 0
        hier = 0
        while True:
            count = 0
            for n in self.nodes:
                if n.hierarchy == hier:
                    for u in n.upstreams:
                        if u.hierarchy is None:
                            u.hierarchy = hier + 1
                            count += 1
            hier += 1
            if count == 0:
                self.max_hier = hier
                break
    # def get_root_node(self):
    #     node_directions = self.node_directions
