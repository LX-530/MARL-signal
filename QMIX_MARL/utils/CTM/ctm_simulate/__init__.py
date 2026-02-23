"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : MASA-QMIX
@File : ctm_simulate.py
@Author : Chenasuny
@Time : 2024/3/1 22:11
"""
import math
from collections import defaultdict, deque

try:
    from shapely.geometry import Polygon as ShapelyPolygon
except Exception:
    ShapelyPolygon = None

free_velocity = 1.2  # 自由流速度
l = 4
max_density = 6  # 最大密度值 单位：人/米方
cri_density = 2.5  # 临界密度值 单位：人/米方
duration = l / free_velocity  # 时间步长，单位：秒


def polygon_area(points):
    if ShapelyPolygon is not None:
        return float(ShapelyPolygon(points).area)
    if not points:
        return 0.0
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0




class nodeInfo():
    def __init__(self,cell_id,exit_inf, cell_inf, num_inf, fire_info,energy_domine, cell_type): #neighbor_ids
        self.cell_id = cell_id
        self.exit = exit_inf
        self.cell = cell_inf
        self.initial_number = num_inf  # 疏散开始时元胞内行人数
        if not self.exit:
            self.area = polygon_area(cell_inf) * 1000000 # / 100  # 面积
        else:
            self.area = float("inf")
        self.initial_density = float(num_inf) / float(self.area)  # 疏散开始时元胞内行人密度
        self.current_density = self.initial_density
        self.current_number =  self.initial_number
        self.current_fireinfo = fire_info
        self.energy_domine =energy_domine
        self.cell_type =cell_type

    #根据新密度更新数据
    def current_info(self,current_density,current_fire):
        self.current_density = current_density
        if self.exit:
            self.current_number = 0
        else:
            self.current_number =  round(self.current_density * self.area,5)
        self.current_fireinfo = current_fire


class baseGraph():
    def __init__(
        self,
        exit_inf,
        cell_inf,
        init_num_inf,
        adjacency_mat,
        sigal_effect_cells,
        fixed_direction_cell,
        fire_info,
        agent_cell_ids,
        energy_domine,
        cell_type,
        blocked_direction_edges=None,
        forced_direction_edges=None
    ):
        #实例化每一个node
        self.nodesinfo = {}   #键的编号为实际的编号，即以1为开始
        for i in range(len(exit_inf)):
            nodeinfo = nodeInfo(i,exit_inf[i], cell_inf[i], init_num_inf[i],
                                # self.get_neighbors(adjacency_mat,i),
                                self.get_fireinfo(fire_info,i),
                                energy_domine[i+1],
                                cell_type[i])

            self.nodesinfo[i+1] = nodeinfo
        self.exit_info = exit_inf
        self.cell_inf = cell_inf
        self.init_num_inf = init_num_inf
        self.current_num = [0 for _ in init_num_inf]  #一开始初始化所有的元胞数量为0
        self.current_density = [0 for _ in init_num_inf]  # 一开始初始化所有的元胞数量为0
        self.exs_list = []
        self.am = adjacency_mat  # 无向图的邻接矩阵
        self.fixed_direction_cell = fixed_direction_cell
        self.blocked_direction_edges = set(blocked_direction_edges or [])
        self.forced_direction_edges = list(forced_direction_edges or [])
        self.sigal_effect_cells = sigal_effect_cells
        self.signal_available_direction = self.get_signal_available_direction()  # 每个signal可以选择的方向
        self.all_direction_cells = []
        self.get_group_adjacency_matrix = []
        self.cell_number = len(self.nodesinfo)
        self.groups_ids = []  # 对应真实的ID-1，即以0为开始
        self.sub_adj_matrixs = []
        self.groups_directions =[]   ## 对应真实的ID，即以1为开始
        self.fire_info_all = fire_info
        self.fire_info_current = fire_info[0]
        self.agent_cell_ids = agent_cell_ids
        self.energy_domine = energy_domine
        # Preserve the initial global distances as a fallback.
        self._global_energy_domine = dict(energy_domine)
        self.cell_type = cell_type


    def get_signal_available_direction(self):
        signal_available_directions = []
        for key in self.sigal_effect_cells:
            signal_available_directions.append(list(key.keys()))
        return signal_available_directions


    def get_neighbors(self, adj_matrix, node):
        """
        返回指定节点的所有相邻节点及其接触长度
        :param adj_matrix: 邻接矩阵，二维列表
        :param node: 指定的节点编号（从0开始）
        """
        neighbors = {1:[],2:[],3:[],4:[]}
        for i, contact_info in enumerate(adj_matrix[node]):
            contact_direction = contact_info
            if contact_direction >= 0.1:  # 如果接触长度不为0，则节点i是一个相邻节点
                neighbors[contact_direction] =  neighbors[contact_direction] + [(i+1, contact_info[1])]
        return neighbors

    def get_fireinfo(self,fire_info,i):
        """
        初始化的火灾信息
        """
        return  [fire_info[0][0][i],fire_info[0][1][i],fire_info[0][2][i],fire_info[0][3][i]]


    def from_actions_get_groupId_submatrix(self, actions):
        """
        从动作中获得方向对
        """
        self.all_direction_cells = []
        #初始化一个邻接矩阵
        directions = []
        for i, action in enumerate(actions):
            action_dict = self.sigal_effect_cells[i]
            direction = action_dict[int(action) + 1]
            directions.extend(direction)
        all_direction_cells = directions + self.fixed_direction_cell

        # Apply global edge rules after action composition.
        if self.blocked_direction_edges:
            all_direction_cells = [e for e in all_direction_cells if e not in self.blocked_direction_edges]
        if self.forced_direction_edges:
            for edge in self.forced_direction_edges:
                if edge not in all_direction_cells:
                    all_direction_cells.append(edge)
                reverse_edge = (edge[1], edge[0])
                if reverse_edge in all_direction_cells and reverse_edge not in self.forced_direction_edges:
                    all_direction_cells = [e for e in all_direction_cells if e != reverse_edge]

        remove_set = set()
        # 遍历所有方向对
        for i_direction in all_direction_cells:
            # 如果这个方向对已经在移除列表中，则跳过
            if i_direction in remove_set:
                continue
            x1, x2 = i_direction
            reversed_direction = (x2, x1)

            # 检查反向对是否存在于all_direction_cells中
            if reversed_direction in all_direction_cells:
                # 将双向对都加入到移除集合中
                remove_set.add(i_direction)
                remove_set.add(reversed_direction)
        filtered_direction_cells = [direction for direction in all_direction_cells if direction not in remove_set]

        self.all_direction_cells = filtered_direction_cells

        # 下面得到子图的邻接矩阵
        self.directions2matrix() #
        self.form_adj_matrix_get_subgroup_ids() #
        self.from_subgroup2directions()  # 获得group的方向对
        self.subgroup_ids2sub_adj_matrix()
        self.update_energy_domine_by_groups()



    def directions2matrix(self):

        adjacency_matrix = [
            [0] * self.cell_number for _ in range(self.cell_number)
        ]
        # 根据提供的边在邻接矩阵中标记相应的位置
        for edge in self.all_direction_cells:
            start_node, end_node = edge
            adjacency_matrix[start_node-1][end_node-1] = 1
        self.get_group_adjacency_matrix = adjacency_matrix


    def form_adj_matrix_get_subgroup_ids(self):
        """
        从邻接矩阵中得到子图IDs
        """
        groups = {}
        conbining_group = {}

        for i in range(len(self.get_group_adjacency_matrix)):
            node_neighbor = self.get_group_adjacency_matrix[i]
            indices_list = [i for i, x in enumerate(node_neighbor) if x == 1]
            indices = set(indices_list)
            for key, value in groups.items():
                intersection = value & set(indices)
                if i in value or intersection:
                    value_new = value | set(indices) | {i, }
                    groups[key] = value_new
                    indices = {}
            if indices != {}:
                value_new = indices | {i, }
                groups[len(groups)] = value_new

        for key, value in conbining_group.items():
            combine_item = list(value)[0]
            combine_dict = {}
            value_in_combining_group = groups[combine_item]
            combine_dict[combine_item] = value_in_combining_group
            for i in value:
                value_in_combining_group = groups.pop(i)
                combine_dict[combine_item] = combine_dict[combine_item] | value_in_combining_group
            groups.update(combine_dict)
        ids_set = {}
        for value in groups.values():
            ids_set = set(ids_set) | value
        if len(ids_set) < len(self.get_group_adjacency_matrix):
            for i in range(len(self.get_group_adjacency_matrix)):
                intersection = ids_set & set(i)
                if intersection == {}:
                    groups[len(groups) + len(self.get_group_adjacency_matrix)] = {i, }

        groups_ids = [list(xx) for xx in list(groups.values())]


        removal_repeat_groups = []
        list_group = []
        #再一次去重复
        for i , igroup in enumerate(groups_ids):
            for j , jgroup in enumerate(groups_ids):
                intset = set(igroup) & set(jgroup)
                if i != j and len(intset)>0:
                    removal_repeat_groups.append([i,j])
                    list_group.append(i)
                    list_group.append(j)

        finianl_groups_ids = []
        for ii in removal_repeat_groups:
            merge = set(groups_ids[ii[0]]) | set(groups_ids[ii[1]])
            finianl_groups_ids.append(list(merge))
        for jj,group  in enumerate( groups_ids):
            if jj not in list_group:
                finianl_groups_ids.append(group)

        def merge_overlapping_sublists(two_d_list):
            parent = {}
            size = {}
            element_to_index = defaultdict(list)

            # 初始化并查集
            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x, y):
                rootX = find(x)
                rootY = find(y)
                if rootX != rootY:
                    if size[rootX] > size[rootY]:
                        parent[rootY] = rootX
                        size[rootX] += size[rootY]
                    else:
                        parent[rootX] = rootY
                        size[rootY] += size[rootX]

            # 建立映射关系，记录每个元素所属的子列表索引
            for i, sublist in enumerate(two_d_list):
                parent[i] = i
                size[i] = 1
                for elem in sublist:
                    element_to_index[elem].append(i)

            # 对于每个元素，将其所有出现的子列表索引进行合并
            for indices in element_to_index.values():
                for i in range(1, len(indices)):
                    union(indices[i - 1], indices[i])

            # 根据并查集的结果构建最终的合并后的子列表
            merged = defaultdict(set)
            for i, sublist in enumerate(two_d_list):
                root = find(i)
                merged[root].update(sublist)

            # 将结果转换为列表形式返回
            return [list(s) for s in merged.values()]

        self.groups_ids = merge_overlapping_sublists(finianl_groups_ids)


    # def from_subgroup2directions(self):
    #     """
    #     从group的Id中得到有效的方向对
    #     """
    #     all_directions_cells = self.all_direction_cells
    #     groups_directions = []
    #     for group in self.groups_ids:
    #         group_directions = []
    #         for i in group:
    #             direction = [xy for xy in all_directions_cells if i+1 in xy]
    #             group_directions.extend(direction)
    #         unique_list = []
    #         for item in group_directions:
    #             if item not in unique_list:
    #                 unique_list.append(item)
    #         groups_directions.append(unique_list)
    #     self.groups_directions = groups_directions

    def from_subgroup2directions(self):
        """
        从group的Id中得到有效的方向对，并检测是否存在环路
        """
        all_directions_cells = self.all_direction_cells
        groups_directions = []

        for group in self.groups_ids:
            group_directions = []
            for i in group:
                direction = [xy for xy in all_directions_cells if (i + 1) in xy]
                group_directions.extend(direction)
            unique_list = []
            for item in group_directions:
                if item not in unique_list:
                    unique_list.append(item)
            groups_directions.append(unique_list)
        #去除环路
        loop_detected = False
        edge_to_remove = None

        for idx, group_edges in enumerate(groups_directions):
            source_nodes = set()
            target_nodes = set()
            for edge in group_edges:
                source_nodes.add(edge[0])
                target_nodes.add(edge[1])

            possible_root_nodes = target_nodes - source_nodes
            if not possible_root_nodes:  # 存在环路
                loop_detected = True
                loop_nodes = source_nodes & target_nodes
                if loop_nodes:
                    # 选择第一个环上节点对应的边删除
                    loop_node = next(iter(loop_nodes))
                    for edge in group_edges:
                        if edge[0] == loop_node or edge[1] == loop_node:
                            edge_to_remove = edge
                            break
                    break  # 只处理第一个发现的环路

        if loop_detected and edge_to_remove:
            if edge_to_remove in self.all_direction_cells:
                self.all_direction_cells.remove(edge_to_remove)
            # 重新计算结构
            self.directions2matrix()
            self.form_adj_matrix_get_subgroup_ids()
            self.from_subgroup2directions()  # 递归调用自身再次检查
        else:
            self.groups_directions = groups_directions

    def subgroup_ids2sub_adj_matrix(self):
        """
        """
        sub_am =[]
        groups_direction = self.groups_directions
        for j, group_direction in enumerate(groups_direction):
            group_id = self.groups_ids[j]
            adj_matrix = [[0] * len(group_id) for _ in range(len(group_id))] #初始化一个新的邻接矩阵
            for x,y in group_direction:
                adj_matrix[group_id.index(x-1)][group_id.index(y-1)] = self.am[x-1][y-1]
            sub_am.append(adj_matrix)
        self.sub_adj_matrixs = sub_am

    def _group_shortest_steps_to_exits(self, group_id_list, sub_adj_matrix):
        """
        Compute shortest steps (unit weight) from each node in a subgroup to the nearest exit,
        using the directed subgroup adjacency matrix. Returns a dict keyed by global node id (1-based).
        """
        exit_locals = [i for i, gid in enumerate(group_id_list) if self.exit_info[gid]]
        if not exit_locals:
            return {}

        n = len(sub_adj_matrix)
        rev_adj = [[] for _ in range(n)]
        for u in range(n):
            row = sub_adj_matrix[u]
            for v, w in enumerate(row):
                if w and w > 0:
                    rev_adj[v].append(u)

        dist = [None] * n
        dq = deque(exit_locals)
        for s in exit_locals:
            dist[s] = 0

        while dq:
            u = dq.popleft()
            du = dist[u]
            for v in rev_adj[u]:
                if dist[v] is None:
                    dist[v] = du + 1
                    dq.append(v)

        result = {}
        for local_idx, gid in enumerate(group_id_list):
            if dist[local_idx] is not None:
                result[gid + 1] = dist[local_idx]
        return result

    def update_energy_domine_by_groups(self):
        """
        Recompute energy_domine per subgroup using directed edges. Nodes that are not reachable
        to any exit in their subgroup keep the global baseline value.
        """
        if not self.sub_adj_matrixs or not self.groups_ids:
            return

        new_energy = dict(self._global_energy_domine)
        for group_id_list, sub_adj in zip(self.groups_ids, self.sub_adj_matrixs):
            group_energy = self._group_shortest_steps_to_exits(group_id_list, sub_adj)
            if group_energy:
                new_energy.update(group_energy)

        self.energy_domine = new_energy
        for node_id, node in self.nodesinfo.items():
            if node_id in new_energy:
                node.energy_domine = new_energy[node_id]


    def update_node_num_info(self,finished_subgroups, step):
        fire_info_all = self.fire_info_all[step]
        for sub_group in finished_subgroups:
            nodes = sub_group.nodes
            for node in nodes:
                fire_info = [fire_info_all[0][node.real_ID-1],fire_info_all[1][node.real_ID-1],fire_info_all[2][node.real_ID-1],fire_info_all[3][node.real_ID-1]]
                self.nodesinfo[node.real_ID].current_info(node.density,fire_info)
        for id, nd in self.nodesinfo.items():
            self.current_num[id-1] = nd.current_number
            self.current_density[id-1] = nd.current_density
        self.fire_info_current = fire_info_all
# cd D:\github\signal\MARL-signal\QMIX_MARL
# New-Item -ItemType Directory -Force .\.mplconfig | Out-Null
# $env:PYTHONPATH="$PWD"
# $env:MPLCONFIGDIR="$PWD\.mplconfig"
# python -m utils.CTM.ctm_start.__init__
