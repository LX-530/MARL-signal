"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
@Project : MASA-QMIX
@File : ctm_start.py
@Author : Chenasuny
@Time : 2024/4/1 22:30
"""
import csv
import random
from pathlib import Path

from sympy.stats import density

from utils.CTM.ctm_simulate import baseGraph, nodeInfo
from utils.CTM.graph_node import Graph
from utils.CTM.utils import random_people, load_all_firebytime, cell_type_create
import networkx as nx
#import matplotlib.pyplot as plt



import matplotlib
matplotlib.use('Agg')  # 关键修复：禁用交互模式
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from matplotlib.patches import Polygon, FancyArrowPatch
import imageio
import os
from utils.CTM.utils import calculate_fire_risk, load_all_firebytime

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / 'data_preprocessing'
OUTPUT_DIR = BASE_DIR / 'result' / 'ctm_visuals'
VISUAL_OUTPUT_DIR = BASE_DIR / 'result' / 'visual'



free_velocity = 1.2  # 自由流速度
l = 4
max_density = 6  # 最大密度值 单位：人/米方
cri_density = 2.5  # 临界密度值 单位：人/米方
duration = l / free_velocity  # 时间步长，单位：秒


LAYER2_TARGETS = [109, 117, 134, 138, 157]
LAYER3_TARGETS = [167, 170, 176]
LAYER4_TARGETS = [198, 205, 217, 228]
DEFAULT_EXIT_NODES = [100, 101, 102]
HIDDEN_NODES = {103}

def _build_layer_sets(num_nodes):
    # Keep the exact user-requested ranges.
    return {
        "layer1": set(range(1, 105)),
        "layer2": set(range(104, 163)),
        "layer3": set(range(164, 177)),
        "layer4": set(range(177, num_nodes + 1)),
    }


def find_shortest_paths_to_exits(adj_matrix, exits):
    num_nodes = len(adj_matrix)
    exits = exits or DEFAULT_EXIT_NODES

    graph = nx.Graph()
    graph.add_nodes_from(range(1, num_nodes + 1))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i][j] != 0:
                graph.add_edge(i + 1, j + 1, weight=1.0)

    layer_sets = _build_layer_sets(num_nodes)
    shortest_paths = {}

    layer1_nodes = layer_sets["layer1"] & set(graph.nodes)
    layer1_exits = [node for node in exits if node in layer1_nodes]
    if layer1_exits:
        layer1_graph = graph.subgraph(layer1_nodes)
        layer1_lengths = nx.multi_source_dijkstra_path_length(layer1_graph, layer1_exits, weight='weight')
        for node in layer1_nodes:
            shortest_paths[node] = layer1_lengths.get(node, float('inf'))
    else:
        for node in layer1_nodes:
            shortest_paths[node] = float('inf')

    layer_chain = [
        ("layer2", LAYER2_TARGETS, "layer1"),
        ("layer3", LAYER3_TARGETS, "layer2"),
        ("layer4", LAYER4_TARGETS, "layer3"),
    ]

    for layer_name, targets, next_layer_name in layer_chain:
        current_nodes = layer_sets[layer_name] & set(graph.nodes)
        next_nodes = layer_sets[next_layer_name] & set(graph.nodes)
        current_graph = graph.subgraph(current_nodes)

        target_entry_cost = {}
        for target in targets:
            best_cost = float('inf')
            if target in graph:
                for neighbor in graph.neighbors(target):
                    if neighbor not in next_nodes:
                        continue
                    neighbor_cost = shortest_paths.get(neighbor, float('inf'))
                    if neighbor_cost == float('inf'):
                        continue
                    total_cost = float(graph[target][neighbor].get('weight', 1.0)) + neighbor_cost
                    if total_cost < best_cost:
                        best_cost = total_cost
            target_entry_cost[target] = best_cost

        for node in current_nodes:
            best_cost = float('inf')
            for target, entry_cost in target_entry_cost.items():
                if entry_cost == float('inf') or target not in current_graph:
                    continue
                try:
                    local_cost = nx.shortest_path_length(current_graph, source=node, target=target, weight='weight')
                except nx.NetworkXNoPath:
                    continue
                total_cost = local_cost + entry_cost
                if total_cost < best_cost:
                    best_cost = total_cost
            shortest_paths[node] = best_cost

    valid_exits = [node for node in exits if node in graph.nodes]
    global_lengths = nx.multi_source_dijkstra_path_length(graph, valid_exits, weight='weight') if valid_exits else {}
    for node in range(1, num_nodes + 1):
        if node not in shortest_paths or shortest_paths[node] == float('inf'):
            shortest_paths[node] = global_lengths.get(node, float('inf'))

    return shortest_paths

def change_door(original_door_size, no1, no2, number):
    original_door_size[no1 - 1][no2 - 1] = number
    original_door_size[no2 - 1][no1 - 1] = number

def change_door_1(original_door_size, no1, no2, number):
    original_door_size[no1][no2] = number
    original_door_size[no2][no1] = number



def init(fire_info=None):
    """
    fireinfo : 火灾的信息
    agent_ids : 智能体说处在的元胞的ID
    """

    people_number = random_people()
    cell_type = cell_type_create()

    sigal_effect_cells = [{1: [(1, 2), (2, 3), (3, 4), (66, 65), (67, 66), (68, 67), (69, 70), (70, 71), (71, 117)],
                           2: [(1, 100), (2, 1), (3, 2), (66, 67), (67, 2), (68, 1), (69, 68), (70, 69), (71, 70)],
                           4: [(1, 68), (2, 1), (3, 2), (66, 67), (67, 68), (68, 69)]},  # agent1
                          {1: [(9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (56, 55), (57, 56), (58, 57), (59, 58), (60, 59)],
                           3: [(9, 8), (10, 9), (11, 10), (12, 11), (13, 12), (14, 13), (56, 57), (57, 58), (58, 59), (59, 60), (60, 61)]},  # agent2
                          {1: [(19, 20), (20, 39), (39, 40), (48, 47), (49, 48)],
                           2: [(21, 34), (22, 33), (23, 24), (34, 35)],
                           3: [(19, 18), (20, 19), (39, 20), (48, 49), (49, 50)],
                           4: [(19, 20), (20, 39), (21, 20), (22, 21), (23, 22), (34, 39), (39, 40), (48, 47), (49, 48)]},  # agent3
                          {2: [(26, 27), (27, 102), (28, 103), (29, 28), (30, 27), (31, 30), (36, 37), (37, 30), (38, 29)],
                           4: [(26, 31), (27, 30), (28, 29), (29, 38), (30, 37), (31, 36), (36, 41), (37, 42), (38, 43)]},  # agent4
                          {1: [(40, 41)],
                           2: [(24, 25), (25, 26), (32, 31), (33, 32), (35, 36)],
                           3: [(40, 39)],
                           4: [(24, 33), (25, 32), (32, 35), (33, 34), (35, 40), (40, 41)]},  # agent5
                          {1: [(41, 42), (42, 37), (43, 38), (44, 43), (45, 42), (46, 45), (47, 46)],
                           2: [(72, 46), (73, 72), (74, 73)],
                           3: [(41, 40), (42, 41), (43, 42), (44, 45), (45, 46), (46, 47), (47, 48)],
                           4: [(41, 46), (42, 41), (43, 42), (44, 45), (45, 46), (46, 72), (47, 46), (72, 73), (73, 74), (74, 75)]},  # agent6
                          {1: [(15, 16), (16, 17), (17, 18), (18, 19), (50, 49), (51, 50), (52, 51), (53, 52), (54, 53), (55, 54)],
                           3: [(15, 14), (16, 15), (17, 16), (18, 17), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56)],
                           4: [(18, 19), (50, 49), (51, 50)]},  # agent7
                          {1: [(4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (61, 60), (62, 61), (63, 62), (64, 63), (65, 64), (104, 105)],
                           2: [(104, 117)],
                           3: [(4, 3), (5, 4), (6, 5), (7, 6), (8, 7), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66)]},  # agent8
                          {1: [(82, 81)],
                           2: [(75, 74), (76, 77), (77, 78), (78, 79), (79, 80), (80, 81), (81, 82)],
                           3: [(76, 75), (77, 76), (78, 77), (79, 78), (80, 79), (81, 80)],
                           4: [(75, 76), (82, 83)]},  # agent9
                          {1: [(83, 82), (84, 83), (85, 84), (86, 85), (93, 86), (94, 85), (95, 94)],
                           4: [(83, 84), (84, 85), (85, 101), (86, 85), (93, 86), (94, 85), (95, 94)]},  # agent10
                          {1: [(87, 86), (88, 87), (89, 88), (90, 89), (91, 88), (92, 87), (96, 93), (97, 92), (98, 91), (99, 90)],
                           4: [(87, 86), (88, 87), (89, 88), (90, 89), (91, 88), (92, 87), (96, 93), (97, 92), (98, 91), (99, 90)]},  # agent11
                          {1: [(107, 114), (108, 113), (110, 111), (111, 112), (112, 118), (113, 112), (118, 119), (119, 121), (120, 119)],
                           2: [(107, 108), (108, 109), (109, 63), (110, 109), (111, 110), (112, 111), (113, 108)],
                           3: [(107, 106), (108, 107), (110, 111), (111, 108), (112, 113), (113, 114), (118, 112), (119, 118), (120, 119)],
                           4: [(109, 108)]},  # agent12
                          {1: [(105, 106), (106, 107), (114, 113), (115, 114), (116, 115), (117, 116)],
                           2: [(105, 104), (106, 105), (114, 107), (115, 116), (116, 117), (117, 71)],
                           3: [(114, 115)]},  # agent13
                          {1: [(121, 122), (122, 127), (123, 126), (124, 125), (125, 130), (126, 129), (127, 128), (129, 132), (130, 131)],
                           3: [(121, 119), (122, 121), (123, 122), (124, 123), (125, 126), (126, 123), (127, 122), (129, 126), (130, 125)],
                           4: [(125, 126), (126, 127), (127, 128), (129, 132), (130, 129)]},  # agent14
                          {1: [(128, 133), (131, 136), (132, 135), (133, 134), (134, 138), (135, 137), (136, 135), (137, 140), (138, 139), (139, 142), (140, 141)],
                           3: [(128, 127), (131, 130), (132, 129), (133, 128), (134, 133), (135, 132), (136, 131), (137, 135), (138, 134), (139, 138), (140, 137)],
                           4: [(128, 133), (131, 132), (132, 135), (133, 134), (134, 97), (135, 134), (136, 135), (137, 138), (138, 96), (139, 138), (140, 137)]},  # agent15
                          {1: [(141, 146), (142, 145), (146, 149), (147, 148), (148, 157), (149, 156), (156, 159), (157, 158), (158, 159), (159, 73)],
                           2: [(146, 149), (147, 148), (148, 157), (149, 156), (156, 157), (157, 48), (158, 157), (159, 158)],
                           3: [(141, 140), (142, 139), (146, 141), (147, 146), (148, 149), (149, 146), (156, 149), (157, 156), (158, 159), (159, 156)],
                           4: [(141, 140), (142, 139)]},  # agent16
                          {1: [(143, 144), (144, 151), (145, 150), (150, 155), (151, 154), (152, 153), (153, 154), (154, 161), (155, 160), (160, 159), (161, 160), (162, 161)],
                           2: [(143, 144), (144, 151), (145, 150), (150, 155), (151, 154), (152, 153), (153, 154), (154, 155), (155, 156), (160, 159), (161, 160), (162, 161)],
                           3: [(143, 142), (144, 143), (145, 142), (150, 145), (151, 144), (152, 153), (153, 154), (154, 151), (155, 150), (160, 155), (161, 154), (162, 161)]},  # agent17
                          {1: [(187, 188), (188, 193), (193, 194), (194, 199), (199, 200), (200, 201), (201, 202), (195, 198), (198, 164)],
                           4: [(187, 188), (188, 193), (193, 194), (194, 195), (199, 198), (200, 199), (201, 200), (195, 198), (198, 164)]},  # agent18
                          {1: [(185, 190), (190, 191), (191, 196), (196, 197), (197, 210), (209, 208), (210, 209)],
                           2: [(185, 190), (190, 191), (191, 196), (196, 195), (197, 198), (209, 210), (210, 197)]},  # agent19
                          {1: [(202, 203), (203, 204), (204, 223), (221, 220), (222, 221), (223, 222), (205, 168)],
                           3: [(202, 201), (203, 202), (204, 203), (221, 222), (222, 223), (223, 204), (205, 168)],
                           4: [(202, 203), (203, 204), (204, 205), (221, 222), (222, 223), (223, 204), (205, 168)]},  # agent20
                          {1: [(206, 211), (207, 206), (208, 207), (211, 212), (212, 213), (213, 214)],
                           2: [(206, 205), (207, 206), (208, 207), (211, 206), (212, 211), (213, 212)],
                           3: [(206, 207), (207, 208), (208, 209), (211, 206), (212, 211), (213, 212)]},  # agent21
                          {1: [(218, 217), (219, 218), (220, 219), (224, 225), (229, 228)],
                           3: [(218, 219), (219, 220), (220, 221), (224, 218), (229, 224)]},  # agent22
                          {1: [(217, 225), (225, 228), (228, 173)],
                           2: [(217, 172), (225, 228), (228, 173)],
                           3: [(217, 172), (225, 217), (228, 173)],
                           4: [(217, 172), (225, 217), (228, 225)]},  # agent23
                          {1: [(214, 215), (215, 216), (216, 217), (226, 225), (227, 228)],
                           3: [(214, 213), (215, 214), (216, 215), (226, 216), (227, 226)]},  # agent24
                          ]

    fixed_cell = [(164,165),(165,166),(166,167),(167,109), (168,169),(169,170),(172,171),(171,170),(170,120), (173,174),(174,175),(175,176),(176,147) #一楼
                             ,(177,178),(178,179),(179,180),(180,187),(181,182),(182,183),(183,184),(184,185),  (186, 189), (189, 192), (192, 195), (195, 198), (198, 164), (205, 168)
                  ,(235,234),(234,233),(233,232),(232,231),(231,230),(230,229),  (236,234),   (237,238),(238,239),(239,240),(240,241),(241,242), (242,227)
                  ,(243,183),(244,184),(245,185),(246,190),(247,191),(248,196),(249,197),(250,210),(251,209),(252,208),(253,207),(254,206),(255,211),(256,212),(257,213),(258,214),(259,215),(260,216),(261,226),(262,227),(263,242),(264,241),(265,240),(266,239),

                  (68, 1), (2, 1), (3, 2), (1, 100), (66, 67), (67, 2), (9, 8), (8, 7), (7, 6), (6, 5), (5, 4), (4, 3),
                  (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (109, 63),
                  (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (59, 58), (58, 57),
                  (57, 56), (56, 55), (55, 54), (54, 53), (53, 52), (52, 51), (99, 90), (90, 89), (98, 91), (91, 88),
                  (97, 92), (92, 87), (96, 93), (93, 86), (95, 94), (94, 85), (89, 88), (88, 87), (87, 86), (86, 85),
                  (85, 101), (84, 85), (83, 84), (82, 83),
                  (18, 19), (19, 20), (20, 39), (39, 40), (40, 41), (41, 42), (42, 37), (43, 38), (51, 50), (50, 49),
                  (49, 48), (48, 47), (47, 46), (46, 45), (45, 42), (44, 43), (23, 24), (24, 25), (25, 26), (26, 27),
                  (22, 33), (33, 32), (32, 31), (31, 30), (21, 34), (34, 35), (35, 36), (36, 37), (37, 30), (30, 27),
                  (27, 102), (38, 29), (29, 28),
                  (75, 74), (74, 73), (73, 72), (72, 46), (81, 80), (80, 79), (79, 78), (78, 77), (77, 76), (76, 75)

                  ]

    # agent_cell_ids = [32, 43, 46, 50, 51, 57, 60, 62, 18, 13, 11, 10, 8, 5, 25, 17, 30, 66] #15,

    # agent_cell_ids = [71,113,119,134,156,194,196,204,206,224,225,226]
    agent_cell_ids = [1, 11, 21, 30, 32, 45, 52, 63, 80, 85, 91, 111, 116, 123, 134, 148, 154, 194, 196, 204, 206, 224, 225, 226]
    # energy_domine = [8,7,6,5,4,5,6,7,7,6,5,4,3,2,1,2,3,4,5,6,7,8,9,3,2,1,2,3,4,5,6,7,8,9,10,11,12,13,8,9,10,11
                     # ,12,13,14,15,16,16,15,14,13,14,15,16,17,12,11,10,9,8,7,6,5,5,4,3,2,1,0,0,0]




    # Read CSV file
    original_door_size = []
    with (DATA_DIR / 'outputL.csv').open('r') as file:
        reader = csv.reader(file)
        for row in reader:
            original_door_size.append([int(float(cell)) for cell in row])  # Convert each cell to integer

    # Manually modify adjacency matrix values
    node_203_index = 203 - 1
    node_204_index = 204 - 1
    node_199_index = 199 - 1
    node_200_index = 200 - 1
    node_184_index = 184 - 1
    node_185_index = 185 - 1

    original_door_size[node_203_index][node_204_index] = 3.45
    original_door_size[node_204_index][node_203_index] = 3.45
    original_door_size[node_199_index][node_200_index] = 3.45
    original_door_size[node_200_index][node_199_index] = 3.45
    original_door_size[node_184_index][node_185_index] = 2
    original_door_size[node_185_index][node_184_index] = 2

    change_door(original_door_size, 164, 165, 3.1)
    change_door(original_door_size, 165, 166, 3.1)
    change_door(original_door_size, 167, 166, 3.1)
    change_door(original_door_size, 167, 109, 3.1)
    change_door(original_door_size, 198, 164, 3.1)

    change_door(original_door_size, 177, 178, 2)
    change_door(original_door_size, 178, 179, 2)
    change_door(original_door_size, 179, 180, 2)
    change_door(original_door_size, 180, 187, 2)

    change_door(original_door_size, 181, 182, 2)
    change_door(original_door_size, 182, 183, 2)
    change_door(original_door_size, 183, 184, 2)
    change_door(original_door_size, 184, 185, 2)

    change_door(original_door_size, 199, 200, 3.45)
    change_door(original_door_size, 200, 201, 3.45)
    change_door(original_door_size, 201, 202, 3.45)
    change_door(original_door_size, 202, 203, 3.45)
    change_door(original_door_size, 203, 204, 3.45)

    change_door(original_door_size, 205, 168, 1.6)
    change_door(original_door_size, 168, 169, 1.6)
    change_door(original_door_size, 169, 170, 1.6)
    change_door(original_door_size, 217, 172, 1.6)
    change_door(original_door_size, 172, 171, 1.6)
    change_door(original_door_size, 171, 170, 1.6)
    change_door(original_door_size, 170, 120, 4)

    change_door(original_door_size, 228, 173, 2)
    change_door(original_door_size, 173, 174, 2)
    change_door(original_door_size, 174, 175, 2)
    change_door(original_door_size, 175, 176, 2)
    change_door(original_door_size, 176, 147, 2)

    change_door(original_door_size, 71, 117, 2.6)
    change_door(original_door_size, 109, 63, 3.25)
    change_door(original_door_size, 157, 48, 3.25)
    change_door(original_door_size, 134, 97, 3.25)
    change_door(original_door_size, 138, 96, 3.25)
    change_door(original_door_size, 159, 73, 4)

    change_door(original_door_size, 214, 215, 3)
    change_door(original_door_size, 213, 214, 3)

    change_door_1(original_door_size, 48, 157, 0)
    change_door_1(original_door_size, 63, 109, 0)
    change_door_1(original_door_size, 71, 117, 0)
    change_door_1(original_door_size, 73, 159, 0)
    change_door_1(original_door_size, 96, 138, 0)
    change_door_1(original_door_size, 97, 137, 0)
    change_door_1(original_door_size, 101, 102, 0)
    change_door_1(original_door_size, 109, 167, 0)
    change_door_1(original_door_size, 120, 170, 0)
    change_door_1(original_door_size, 147, 176, 0)
    change_door_1(original_door_size, 168, 205, 0)
    change_door_1(original_door_size, 170, 120, 0)
    change_door_1(original_door_size, 172, 217, 0)
    change_door_1(original_door_size, 173, 228, 0)
    change_door_1(original_door_size, 174, 146, 0)
    change_door_1(original_door_size, 198, 164, 0)

    change_door(original_door_size, 183, 243, 1.4)
    change_door(original_door_size, 184, 244, 1.4)
    change_door(original_door_size, 185, 245, 1.4)
    change_door(original_door_size, 190, 246, 1.4)
    change_door(original_door_size, 191, 247, 1.4)
    change_door(original_door_size, 196, 248, 1.4)
    change_door(original_door_size, 197, 249, 1.4)
    change_door(original_door_size, 210, 250, 1.4)
    change_door(original_door_size, 209, 251, 1.4)
    change_door(original_door_size, 208, 252, 1.4)
    change_door(original_door_size, 207, 253, 1.4)
    change_door(original_door_size, 206, 254, 1.4)
    change_door(original_door_size, 211, 255, 1.4)
    change_door(original_door_size, 212, 256, 1.4)
    change_door(original_door_size, 213, 257, 1.4)
    change_door(original_door_size, 214, 258, 1.4)
    change_door(original_door_size, 215, 259, 1.4)
    change_door(original_door_size, 216, 260, 1.4)
    change_door(original_door_size, 226, 261, 1.4)
    change_door(original_door_size, 227, 262, 1.4)
    change_door(original_door_size, 242, 263, 1.4)
    change_door(original_door_size, 241, 264, 1.4)
    change_door(original_door_size, 240, 265, 1.4)
    change_door(original_door_size, 239, 266, 1.4)

    # Remove exit 103 entirely: detach all its graph relations.
    exit_103_idx = 103 - 1
    for idx in range(len(original_door_size)):
        original_door_size[exit_103_idx][idx] = 0
        original_door_size[idx][exit_103_idx] = 0

    exits = [100, 101, 102]  #出口的编号

    energy_domine = find_shortest_paths_to_exits(original_door_size, exits)
    shortest_path_arrows = []
    path_graph = nx.Graph()
    path_graph.add_nodes_from(range(1, len(original_door_size) + 1))
    for i in range(len(original_door_size)):
        for j in range(i + 1, len(original_door_size)):
            if original_door_size[i][j] != 0:
                path_graph.add_edge(i + 1, j + 1, weight=1.0)

    layer_sets = _build_layer_sets(len(original_door_size))
    layer_targets = {
        "layer1": set(exits),
        "layer2": set(LAYER2_TARGETS),
        "layer3": set(LAYER3_TARGETS),
        "layer4": set(LAYER4_TARGETS),
    }
    cross_layer_next = {
        "layer4": "layer3",
        "layer3": "layer2",
        "layer2": "layer1",
    }

    def get_layer(node_id):
        # Keep layer priority consistent with shortest-path assignment.
        if node_id in layer_sets["layer1"]:
            return "layer1"
        if node_id in layer_sets["layer2"]:
            return "layer2"
        if node_id in layer_sets["layer3"]:
            return "layer3"
        if node_id in layer_sets["layer4"]:
            return "layer4"
        return None

    layer_closest_dist = {}
    layer_assigned_target = {}
    layer_target_lengths = {}
    for layer_name, nodes in layer_sets.items():
        valid_nodes = nodes & set(path_graph.nodes)
        valid_targets = layer_targets[layer_name] & valid_nodes
        sub_graph = path_graph.subgraph(valid_nodes)
        per_target_lengths = {}
        for target in valid_targets:
            per_target_lengths[target] = nx.single_source_dijkstra_path_length(sub_graph, target, weight='weight')
        layer_target_lengths[layer_name] = per_target_lengths

        closest_dist = {}
        assigned_target = {}
        for node in valid_nodes:
            best = (float('inf'), float('inf'))
            best_target = None
            for target, lengths in per_target_lengths.items():
                dist = lengths.get(node, float('inf'))
                if (dist, target) < best:
                    best = (dist, target)
                    best_target = target
            closest_dist[node] = best[0]
            assigned_target[node] = best_target
        layer_closest_dist[layer_name] = closest_dist
        layer_assigned_target[layer_name] = assigned_target

    for node in sorted(path_graph.nodes):
        node_layer = get_layer(node)
        if node_layer is None:
            continue

        # If already at this layer's target, cross to next layer (if exists).
        if node_layer in cross_layer_next and node in layer_targets[node_layer]:
            next_layer = cross_layer_next[node_layer]
            candidates = []
            for neighbor in path_graph.neighbors(node):
                if get_layer(neighbor) != next_layer:
                    continue
                rank = (
                    layer_closest_dist.get(next_layer, {}).get(neighbor, float('inf')),
                    energy_domine.get(neighbor, float('inf')),
                    neighbor,
                )
                candidates.append((rank, neighbor))
            if candidates:
                shortest_path_arrows.append((node, min(candidates)[1]))
            continue

        # Otherwise, always move within current layer toward nearest target of this layer.
        target = layer_assigned_target.get(node_layer, {}).get(node)
        if target is None:
            continue
        current_dist = layer_target_lengths[node_layer][target].get(node, float('inf'))
        candidates = []
        for neighbor in path_graph.neighbors(node):
            if get_layer(neighbor) != node_layer:
                continue
            neighbor_dist = layer_target_lengths[node_layer][target].get(neighbor, float('inf'))
            if neighbor_dist < current_dist:
                rank = (
                    neighbor_dist,
                    0 if neighbor in layer_targets[node_layer] else 1,
                    energy_domine.get(neighbor, float('inf')),
                    neighbor,
                )
                candidates.append((rank, neighbor))
        if candidates:
            shortest_path_arrows.append((node, min(candidates)[1]))
    shortest_path_arrows = [shortest_path_arrows]
    # Make shortest-path arrows effective in simulation flow, not only visualization.
    fixed_cell = list(dict.fromkeys(fixed_cell + shortest_path_arrows[0]))

    # Enforce one-controller-per-cell for signal effects:
    # each source cell follows its nearest agent only.
    original_control_source_cells = _collect_signal_source_cells(sigal_effect_cells)
    nearest_agent_owner = assign_cells_to_nearest_agent(
        agent_cell_ids,
        candidate_nodes=original_control_source_cells,
        restrict_same_layer=True,
    )
    sigal_effect_cells = build_nearest_agent_signal_effect_cells(
        agent_cell_ids,
        sigal_effect_cells,
        node_owner=nearest_agent_owner,
    )


    Position_cells =[]

    with (DATA_DIR / 'outputP.csv').open('r') as file:
        reader = csv.reader(file)
        for row in reader:
            list1 = []
            for row1 in row:
                list1.append(eval(row1))
            Position_cells.append(list1)


    TF_exit = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    TF_exit[103 - 1] = False

    graph = baseGraph(
        TF_exit,
     Position_cells,
           people_number,
     original_door_size,
        sigal_effect_cells,
        fixed_cell,
     fire_info,
     agent_cell_ids,
     energy_domine,
    cell_type
    )
    graph.shortest_path_arrows = shortest_path_arrows
    graph.original_control_source_cells = sorted(original_control_source_cells)
    graph.nearest_agent_owner = nearest_agent_owner
    graph.nearest_agent_control_cells = build_agent_control_cells(
        agent_cell_ids,
        node_owner=nearest_agent_owner
    )
    return graph


#执行一次动作,即启动CTM一次
def start_Sub_CTM(baseGraph, step):
    all_groups_CTM_finished = []
    for group_idx, adj_matrixs in enumerate(baseGraph.sub_adj_matrixs):
        if len(adj_matrixs) == 1:
            continue
        else:
            group_id_list = baseGraph.groups_ids[group_idx]
            exit_info =  [baseGraph.exit_info[i] for i in group_id_list]
            cell_info =  [baseGraph.cell_inf[i] for i in group_id_list]
            cell_type =  [baseGraph.cell_type[i] for i in group_id_list]
            #这里的number info是变的
            num_info = [baseGraph.current_num[i] for i in group_id_list]
            if step ==1 :
                num_info = [baseGraph.init_num_inf[i] for i in group_id_list]


            fire_info = [[baseGraph.fire_info_current[0][i],baseGraph.fire_info_current[1][i],
                          baseGraph.fire_info_current[2][i],baseGraph.fire_info_current[3][i]] for i in group_id_list]

            group_directions = baseGraph.groups_directions[group_idx]
            sub_graph = Graph(exit_inf=exit_info, cell_inf=cell_info, num_inf=num_info, adjacency_mat=adj_matrixs,
                                         group_id_list=group_id_list, group_directions=group_directions,fire_info=fire_info, cell_type= cell_type)

            for i in range(sub_graph.max_hier):
                for n in sub_graph.nodes:
                    if n.hierarchy == i:
                        n.receive_inflow()
                        n.update_density()

            all_groups_CTM_finished.append(sub_graph)
    # 更新对应的节点info，即人数
    baseGraph.update_node_num_info(all_groups_CTM_finished, step)
    return baseGraph
#
##############################################################################################################################################################################
# ============= 绘制多层连接 =============
def draw_nested_connections(ax, centers, nested_connections):
    """绘制多层连接关系的辅助函数"""
    colors = plt.cm.tab20(np.linspace(0, 1, len(nested_connections)))

    for group_idx, connection_group in enumerate(nested_connections):
        color = colors[group_idx]
        for start, end in connection_group:
            if start in centers and end in centers:
                arrow = FancyArrowPatch(
                    centers[start], centers[end],
                    arrowstyle='->',
                    color=color,
                    mutation_scale=15,
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=2
                )
                ax.add_patch(arrow)
            elif start in HIDDEN_NODES or end in HIDDEN_NODES:
                continue
            else:
                print(f"Warning: Connection {start}->{end} skipped (invalid node)")
    return colors  # 返回颜色用于图例


def _load_cell_centers(csv_path=None):
    """Load polygon centers keyed by node id (1-based)."""
    csv_path = csv_path or (DATA_DIR / 'outputP.csv')

    def parse_coordinate(cell):
        matches = re.findall(r"[-+]?\d*\.?\d+", str(cell))
        return float(matches[0]), float(matches[1])

    df = pd.read_csv(csv_path, header=None, encoding="utf-8")
    centers = {}
    for idx, row in df.iterrows():
        node_id = idx + 1
        pts = []
        for col in range(4):
            x, y = parse_coordinate(row[col])
            pts.append([x, y])
        centers[node_id] = np.mean(pts, axis=0)
    return centers


def _collect_signal_source_cells(sigal_effect_cells):
    """Collect original source cells that are signal-controlled."""
    source_cells = set()
    for action_dict in sigal_effect_cells:
        for edges in action_dict.values():
            for start, _ in edges:
                if start not in HIDDEN_NODES:
                    source_cells.add(start)
    return source_cells


def _node_layer(node_id, layer_sets):
    if node_id in layer_sets["layer1"]:
        return "layer1"
    if node_id in layer_sets["layer2"]:
        return "layer2"
    if node_id in layer_sets["layer3"]:
        return "layer3"
    if node_id in layer_sets["layer4"]:
        return "layer4"
    return None


def assign_cells_to_nearest_agent(agent_cell_ids, candidate_nodes=None, restrict_same_layer=True):
    """Assign each candidate cell to the nearest agent cell by Euclidean distance."""
    centers = _load_cell_centers()
    available_agents = [aid for aid in agent_cell_ids if aid in centers and aid not in HIDDEN_NODES]
    if not available_agents:
        return {}

    if candidate_nodes is None:
        candidate_nodes = [node for node in centers.keys() if node not in HIDDEN_NODES]
    else:
        candidate_nodes = list(candidate_nodes)

    layer_sets = _build_layer_sets(max(centers.keys()))
    agent_layers = {aid: _node_layer(aid, layer_sets) for aid in available_agents}

    owner = {}
    for node in candidate_nodes:
        if node in HIDDEN_NODES or node not in centers:
            continue

        candidate_agents = available_agents
        if restrict_same_layer:
            node_layer = _node_layer(node, layer_sets)
            candidate_agents = [aid for aid in available_agents if agent_layers.get(aid) == node_layer]
            if not candidate_agents:
                continue

        node_xy = centers[node]
        best_rank = (float('inf'), float('inf'))
        best_agent = None
        for agent_id in candidate_agents:
            dist = float(np.linalg.norm(node_xy - centers[agent_id]))
            rank = (dist, agent_id)
            if rank < best_rank:
                best_rank = rank
                best_agent = agent_id
        owner[node] = best_agent
    return owner


def build_nearest_agent_signal_effect_cells(agent_cell_ids, sigal_effect_cells, node_owner=None):
    """
    Keep only edges whose source cell belongs to the same (nearest) agent.
    This enforces one-controller-per-cell for signal-controlled outgoing directions.
    """
    if len(agent_cell_ids) != len(sigal_effect_cells):
        raise ValueError("agent_cell_ids and sigal_effect_cells must have identical lengths")
    if node_owner is None:
        node_owner = assign_cells_to_nearest_agent(agent_cell_ids)

    filtered_all = []
    for idx, agent_id in enumerate(agent_cell_ids):
        action_dict = {}
        for action_key, edges in sigal_effect_cells[idx].items():
            filtered_edges = []
            for start, end in edges:
                if start in HIDDEN_NODES or end in HIDDEN_NODES:
                    continue
                if node_owner.get(start) == agent_id:
                    filtered_edges.append((start, end))
            action_dict[action_key] = filtered_edges
        filtered_all.append(action_dict)
    return filtered_all


def build_agent_control_cells(agent_cell_ids, sigal_effect_cells=None, node_owner=None):
    """Collect controlled cells for each agent."""
    if node_owner is not None:
        control_cells = {agent_id: set() for agent_id in agent_cell_ids}
        for node_id, owner_id in node_owner.items():
            if owner_id in control_cells and node_id not in HIDDEN_NODES:
                control_cells[owner_id].add(node_id)
        return control_cells

    if sigal_effect_cells is None:
        raise ValueError("sigal_effect_cells is required when node_owner is None")
    if len(agent_cell_ids) != len(sigal_effect_cells):
        raise ValueError("agent_cell_ids and sigal_effect_cells must have identical lengths")

    control_cells = {}
    for idx, agent_id in enumerate(agent_cell_ids):
        nodes = set()
        for edges in sigal_effect_cells[idx].values():
            for start, end in edges:
                if start not in HIDDEN_NODES:
                    nodes.add(start)
                if end not in HIDDEN_NODES:
                    nodes.add(end)
        control_cells[agent_id] = nodes
    return control_cells


def _build_distinct_agent_colors(agent_cell_ids):
    """
    Build a high-contrast color map for agent ids.
    Uses multiple qualitative palettes first, then HSV fallback.
    """
    n = max(len(agent_cell_ids), 1)
    palette = []

    qualitative_maps = ["tab20", "tab20b", "tab20c", "Set3", "Dark2", "Paired", "Accent"]
    for cmap_name in qualitative_maps:
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, "colors"):
            palette.extend([tuple(c) for c in cmap.colors])
        else:
            palette.extend([tuple(cmap(i / max(n - 1, 1))) for i in range(n)])

    unique_palette = []
    seen = set()
    for c in palette:
        key = tuple(round(float(x), 3) for x in c[:3])
        if key in seen:
            continue
        seen.add(key)
        unique_palette.append(c)

    if len(unique_palette) < n:
        extra = plt.cm.hsv(np.linspace(0, 1, n - len(unique_palette) + 1, endpoint=False))
        for c in extra:
            key = tuple(round(float(x), 3) for x in c[:3])
            if key in seen:
                continue
            seen.add(key)
            unique_palette.append(tuple(c))
            if len(unique_palette) >= n:
                break

    selected = unique_palette[:n]
    return {agent_id: selected[i] for i, agent_id in enumerate(agent_cell_ids)}


def visualize_agent_control_ranges(agent_cell_ids, sigal_effect_cells=None, exit_nodes=None, output_path=None,
                                   node_owner=None, show_title=False):
    """Visualize agent control ranges with different colors and save to file."""
    csv_path = DATA_DIR / 'outputP.csv'

    def parse_coordinate(cell):
        matches = re.findall(r"[-+]?\d*\.?\d+", str(cell))
        return float(matches[0]), float(matches[1])

    df = pd.read_csv(csv_path, header=None, encoding="utf-8")

    first_poly = [parse_coordinate(df.iloc[0, col]) for col in range(4)]
    dx, dy = -min(p[0] for p in first_poly), -max(p[1] for p in first_poly)

    control_cells = build_agent_control_cells(agent_cell_ids, sigal_effect_cells=sigal_effect_cells, node_owner=node_owner)
    node_to_agents = {}
    for agent_id, nodes in control_cells.items():
        for node in nodes:
            node_to_agents.setdefault(node, []).append(agent_id)

    agent_colors = _build_distinct_agent_colors(agent_cell_ids)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_aspect("equal")
    ax.grid(False)

    centers = {}
    all_points = []
    overlap_exists = False

    for idx, row in df.iterrows():
        node_id = idx + 1
        if node_id in HIDDEN_NODES:
            continue

        poly_points = []
        for col in range(4):
            x, y = parse_coordinate(row[col])
            x, y = x + dx, y + dy
            poly_points.append([x, y])
            all_points.append([x, y])

        owners = node_to_agents.get(node_id, [])
        if exit_nodes and node_id in exit_nodes:
            use_color = "#2ecc71"
            alpha = 0.85
        elif len(owners) == 1:
            use_color = agent_colors[owners[0]]
            alpha = 0.72
        elif len(owners) > 1:
            use_color = "#95a5a6"
            alpha = 0.9
            overlap_exists = True
        else:
            use_color = "#ecf0f1"
            alpha = 0.45

        poly = Polygon(
            poly_points,
            closed=True,
            edgecolor="#2c3e50",
            facecolor=use_color,
            linewidth=0.8,
            alpha=alpha
        )
        ax.add_patch(poly)

        center = np.mean(poly_points, axis=0)
        centers[node_id] = center
        node_label = f"{node_id}*" if len(owners) > 1 else str(node_id)
        ax.text(*center, node_label, fontsize=6, color='black', ha='center', va='center')

    for agent_id in agent_cell_ids:
        if agent_id in centers:
            center = centers[agent_id]
            ax.scatter(
                center[0],
                center[1],
                s=72,
                c=[agent_colors[agent_id]],
                edgecolors="black",
                linewidths=1.0,
                zorder=5
            )

    all_points = np.array(all_points)
    padding = 0.005
    ax.set_xlim(all_points[:, 0].min() - padding, all_points[:, 0].max() + padding)
    ax.set_ylim(all_points[:, 1].min() - padding, all_points[:, 1].max() + padding)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    if show_title:
        ax.set_title("Agent Control Ranges (* means overlap cell)", fontsize=14)

    legend_handles = [
        plt.Line2D(
            [0], [0], marker='s', color='w',
            markerfacecolor=agent_colors[agent_id],
            markeredgecolor='black',
            markersize=9, linestyle='None',
            label=f'Agent {agent_id}'
        )
        for agent_id in agent_cell_ids
    ]
    legend_handles.append(
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#ecf0f1",
                   markeredgecolor='black', markersize=9, linestyle='None', label='No Agent Control')
    )
    if overlap_exists and node_owner is None:
        legend_handles.append(
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#95a5a6",
                       markeredgecolor='black', markersize=9, linestyle='None', label='Overlap Control')
        )
    if exit_nodes:
        legend_handles.append(
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor="#2ecc71",
                       markeredgecolor='black', markersize=9, linestyle='None', label='Exit')
        )
    ax.legend(handles=legend_handles, loc='upper left', fontsize='x-small', ncol=2)

    if output_path is None:
        output_path = VISUAL_OUTPUT_DIR / 'agent_control_ranges.png'
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    return str(output_path)


def visualize_nested_connections(nested_connections, highlight_nodes=None, exit_nodes=None):
    # ============= 默认文件路径 =============
    csv_path = DATA_DIR / 'outputP.csv'

    # ============= 数据准备 =============
    def parse_coordinate(cell):
        matches = re.findall(r"[-+]?\d*\.?\d+", str(cell))
        return float(matches[0]), float(matches[1])

    df = pd.read_csv(csv_path, header=None, encoding="utf-8")

    # ============= 坐标系转换 =============
    first_poly = [parse_coordinate(df.iloc[0, col]) for col in range(4)]
    dx, dy = -min(p[0] for p in first_poly), -max(p[1] for p in first_poly)

    # ============= 可视化设置 =============
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_aspect("equal")
    ax.grid(False)

    centers = {}
    all_points = []

    # ============= 颜色配置 =============
    base_color = "#3498db"  # 默认蓝色
    highlight_color = "#e74c3c"  # 高亮红色
    exit_color = "#2ecc71"  # 出口绿色

    # ============= 绘制所有多边形 =============
    for idx, row in df.iterrows():
        poly_points = []
        node_id = idx + 1
        if node_id in HIDDEN_NODES:
            continue

        # 颜色优先级：出口 > 高亮 > 默认
        if exit_nodes and node_id in exit_nodes:
            use_color = exit_color
        elif highlight_nodes and node_id in highlight_nodes:
            use_color = highlight_color
        else:
            use_color = base_color

        for col in range(4):
            x, y = parse_coordinate(row[col])
            x, y = x + dx, y + dy
            poly_points.append([x, y])
            all_points.append([x, y])

        poly = Polygon(
            poly_points,
            closed=True,
            edgecolor="#2c3e50",
            facecolor=use_color,
            linewidth=1,
            alpha=0.6
        )
        ax.add_patch(poly)

        center = np.mean(poly_points, axis=0)
        centers[node_id] = center

        ax.text(*center, str(node_id),
                fontsize=7, color='white',
                ha='center', va='center')


    # 调用绘制多层连接的函数
    colors = draw_nested_connections(ax, centers, nested_connections)



    # ============= 图形设置 =============
    all_points = np.array(all_points)
    padding = 0.005
    ax.set_xlim(all_points[:, 0].min() - padding, all_points[:, 0].max() + padding)
    ax.set_ylim(all_points[:, 1].min() - padding, all_points[:, 1].max() + padding)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # ============= 图例系统 =============
    legend_patches = []
    # legend_patches = [
    #     plt.Line2D([0], [0], color=colors[i], lw=4, label=f'Group {i + 1}')
    #     for i in range(len(nested_connections))
    # ]

    # 动态添加图例项
    if highlight_nodes:
        legend_patches.append(
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=highlight_color, markersize=10,
                       label='Agents Nodes')
        )
    if exit_nodes:
        legend_patches.append(
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor=exit_color, markersize=10,
                       label='Exit Nodes')
        )

    ax.legend(handles=legend_patches, loc='upper left', fontsize='x-small')

    # ============= 输出结果 =============
    plt.savefig(OUTPUT_DIR / 'nested_connections.png', dpi=300, bbox_inches="tight")
    plt.close()
    return fig



def visualize_numbers(highlight_nodes=None, exit_nodes=None, people_count=None, step=1, nested_connections=None,name='density'):
    # ============= 默认文件路径 =============
    csv_path = DATA_DIR / 'outputP.csv'

    # ============= 数据准备 =============
    def parse_coordinate(cell):
        matches = re.findall(r"[-+]?\d*\.?\d+", str(cell))
        return float(matches[0]), float(matches[1])

    df = pd.read_csv(csv_path, header=None, encoding="utf-8")

    # ============= 坐标系转换 =============
    first_poly = [parse_coordinate(df.iloc[0, col]) for col in range(4)]
    dx, dy = -min(p[0] for p in first_poly), -max(p[1] for p in first_poly)

    # ============= 可视化设置 =============
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_aspect("equal")
    ax.grid(False)

    centers = {}
    all_points = []

    # ============= 颜色配置 =============
    base_cmap = plt.cm.get_cmap('RdYlGn_r')  # 红色到绿色的反转颜色映射

    # 固定密度范围在0到6
    min_count = 0
    if name=='density':
        max_count = max_density
    elif name=='people':
        max_count = 96
    elif name=='fire_levels':
        max_count = 50
    elif name == 'congestion':
        max_count = 96
    else:
        max_count = 140

    norm = plt.Normalize(min_count, max_count)

    # ============= 绘制所有多边形 =============
    for idx, row in df.iterrows():
        poly_points = []
        node_id = idx + 1
        if node_id in HIDDEN_NODES:
            continue

        # 颜色优先级：出口 > 默认
        if exit_nodes and node_id in exit_nodes:
            use_color = "#2ecc71"  # 出口绿色
        else:
            if people_count and idx < len(people_count):
                normalized_count = norm(people_count[idx])
                use_color = base_cmap(normalized_count)
            else:
                use_color = "#3498db"  # 默认蓝色

        for col in range(4):
            x, y = parse_coordinate(row[col])
            x, y = x + dx, y + dy
            poly_points.append([x, y])
            all_points.append([x, y])

        poly = Polygon(
            poly_points,
            closed=True,
            edgecolor="#2c3e50",
            facecolor=use_color,
            linewidth=1,
            alpha=0.6
        )
        ax.add_patch(poly)

        center = np.mean(poly_points, axis=0)
        centers[node_id] = center

        # 更新文本标签为人数
        if people_count and idx < len(people_count):
            ax.text(*center, f"{people_count[idx]:.1f}",
                    fontsize=7, color='black',
                    ha='center', va='center')
        else:
            ax.text(*center, str(node_id),
                    fontsize=7, color='white',
                    ha='center', va='center')

    # 调用绘制多层连接的函数
    colors = draw_nested_connections(ax, centers, nested_connections)

    # # ============= 标记智能体点 =============
    # if highlight_nodes:
    #     for node_id in highlight_nodes:
    #         if node_id in centers:
    #             # 获取归一化后的人数颜色
    #             if people_count and node_id - 1 < len(people_count):
    #                 normalized_count = norm(people_count[node_id - 1])
    #                 agent_color = base_cmap(normalized_count)
    #             else:
    #                 agent_color = "#3498db"  # 默认蓝色
    #
    #             # 更新智能体点的文本标签为人数，字体颜色改为黑色
    #             if people_count and node_id - 1 < len(people_count):
    #                 ax.text(centers[node_id][0], centers[node_id][1], f"{people_count[node_id - 1]:.2f}",
    #                         fontsize=6, color='black',
    #                         ha='center', va='center')

    # ============= 图形设置 =============
    all_points = np.array(all_points)
    padding = 0.005
    ax.set_xlim(all_points[:, 0].min() - padding, all_points[:, 0].max() + padding)
    ax.set_ylim(all_points[:, 1].min() - padding, all_points[:, 1].max() + padding)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # 添加标题
    ax.set_title(f"Step {step}", fontsize=14)

    # ============= 图例系统 =============
    legend_patches = []

    # 动态添加图例项
    if exit_nodes:
        legend_patches.append(
            plt.Line2D([0], [0], marker='s', color='w',
                       markerfacecolor="#2ecc71", markersize=10,
                       label='Exit Nodes')
        )

    # 添加智能体点的图例项
    legend_patches.append(
        plt.Line2D([0], [0], color='black', marker='o', markersize=5,
                   linestyle='None', label='Agent Label')
    )

    ax.legend(handles=legend_patches, loc='upper left', fontsize='x-small')

    # ============= 添加颜色条 =============
    sm = plt.cm.ScalarMappable(cmap=base_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('People Count', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # ============= 输出结果 =============
    output_filename = OUTPUT_DIR / f"{name} number{step}.png"
    plt.savefig(output_filename, dpi=150, bbox_inches="tight")
    plt.close()
    return str(output_filename)


def create_gif(image_files, output_gif):
    images = []
    for file in image_files:
        images.append(imageio.imread(file))
    imageio.mimsave(output_gif, images, duration=0.2)

def get_reward(self):
    self.state4marl= _get_state_list1(self)
    reward = -   sum(self.static_field) -  sum(self.fire_levels) - sum(self.congestion_levels)
    return  reward,sum(self.static_field) ,  sum(self.fire_levels) , sum(self.congestion_levels)

def _get_state_list1(self):
    """
    全局的状态
    """
    all_nodesinfo = self.nodesinfo
    all_people_number = 0
    cell_people_count = []
    #静态场
    static_field = []
    fire_levels = []
    congestion_levels = []
    for i in all_nodesinfo.values():
        N = i.current_number
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
    return [all_people_number] + cc , cell_people_count#


if __name__ == '__main__':
    # Run from repo root or QMIX_MARL; outputs saved under QMIX_MARL/result/ctm_visuals
    # Static actions are fixed to 0 for every signal.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fire_info = load_all_firebytime()
    # action_space = [
    #     [1, 2, 2, 3, 0, 3, 1, 3, 1, 0, 0, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 3, 0],
    #     [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 3, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 3, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 1, 0, 3, 1, 0, 1, 0, 1, 0, 1, 0],
    #     [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    #     [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    #     [1, 1, 2, 2, 1, 0, 1, 0, 1, 0, 1, 0],
    #     [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    #     [1, 1, 2, 2, 1, 0, 1, 2, 1, 2, 1, 2],
    #     [0, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2],
    #     [0, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0],
    #     [0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 2, 2, 3, 2, 3, 0, 3, 2, 2, 3, 2],
    #     [1, 1, 2, 3, 1, 3, 1, 3, 1, 2, 1, 2],
    #     [1, 1, 2, 3, 1, 3, 1, 3, 1, 2, 1, 2],
    #     [1, 1, 2, 2, 1, 0, 1, 2, 1, 2, 1, 2],
    #     [1, 1, 2, 2, 1, 0, 1, 2, 1, 2, 1, 2],
    #     [1, 1, 2, 2, 1, 0, 1, 2, 1, 2, 1, 2],
    #     [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    #     [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    #     [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    #     [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 3, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 3, 0],
    #     [0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0],
    #     [0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0],
    #     [0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0],
    #     [0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0],
    #     [0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0],
    #     [0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0],
    #     [0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0],
    #     [0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0],
    #     [0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0],
    #     [0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0],
    #     [0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0],
    #     [0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 3, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 3, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 3, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 3, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 3, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 3, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 3, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 3, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 1, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 1, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 1, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 1, 0],
    #     [1, 1, 0, 3, 1, 3, 1, 3, 1, 0, 1, 0]
    # ]
    g = init(fire_info)
    agent_control_path = visualize_agent_control_ranges(
        g.agent_cell_ids,
        node_owner=getattr(g, "nearest_agent_owner", None),
        exit_nodes=[100, 101, 102],
        output_path=VISUAL_OUTPUT_DIR / "agent_control_ranges.png",
        show_title=False
    )
    print(f"Saved agent control ranges: {agent_control_path}")
    path_connections = getattr(g, "shortest_path_arrows", g.groups_directions)
    static_actions = [0] * len(g.sigal_effect_cells)
    action_space = [static_actions]

    # 存储图片
    density_image_files = []
    fire_levels_image_files = []
    congestion_image_files = []
    static_image_files = []

    # 全局的level
    static_fields = []
    fire_levels_list = []
    congestion_levels_list = []
    total_people_list = []  # 新增：总人数列表
    iterations = []
    location = []
    rewards = 0
    for i in range(0, 10):
        actions = static_actions
        g.from_actions_get_groupId_submatrix(actions)
        g = start_Sub_CTM(g, i)
        print(sum(g.current_num))
        reward, static_field, fire_levels, congestion_levels = get_reward(g)
        rewards += static_field + fire_levels + congestion_levels
        # location = [g.energy_domine[key] for key in sorted(g.energy_domine.keys())]

        # 记录数据
        static_fields.append(static_field)
        fire_levels_list.append(fire_levels)
        congestion_levels_list.append(congestion_levels)

        # 新增：计算并记录总人数
        total_people = sum(g.current_num.values()) if isinstance(g.current_num, dict) else sum(g.current_num)
        total_people_list.append(total_people)

        iterations.append(i)

        #生成可视化图片
        density_image_files.append(visualize_numbers(g.agent_cell_ids, [100, 101, 102],
                                                     g.current_num, i, path_connections, 'people'))
    #     fire_levels_image_files.append(visualize_numbers(g.agent_cell_ids, [100, 101, 102],
    #                                                      g.fire_levels, i, g.groups_directions, 'fire_levels'))
    #     congestion_image_files.append(visualize_numbers(g.agent_cell_ids, [100, 101, 102],
    #                                                     g.congestion_levels, i, g.groups_directions, 'congestion'))
    #     static_image_files.append(visualize_numbers(g.agent_cell_ids, [100, 101, 102],
    #                                                 g.static_field, i, g.groups_directions, 'static'))
    #     # visualize_numbers(g.agent_cell_ids, [100, 101, 102],
    #     #                                              location, i, g.groups_directions, 'location')
    # #
    # # 1. 静态场变化图
    # plt.figure(figsize=(10, 4))
    # plt.plot(iterations, static_fields, color='blue')
    # plt.title('Static Field over Iterations')
    # plt.xlabel('Time step')
    # plt.ylabel('Static Field')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(OUTPUT_DIR / 'static_field_plot.png')
    # plt.close()

    # # 2. 火灾等级变化图
    # plt.figure(figsize=(10, 4))
    # plt.plot(iterations, fire_levels_list, color='red')
    # plt.title('Fire Levels over Iterations')
    # plt.xlabel('Time step')
    # plt.ylabel('Fire Levels')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(OUTPUT_DIR / 'fire_levels_plot.png')
    # plt.close()

    # # 3. 拥堵等级变化图
    # plt.figure(figsize=(10, 4))
    # plt.plot(iterations, congestion_levels_list, color='green')
    # plt.title('Congestion Levels over Iterations')
    # plt.xlabel('Time step')
    # plt.ylabel('Congestion Levels')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(OUTPUT_DIR / 'congestion_levels_plot.png')
    # plt.close()

    # # 4. 累积火灾等级图
    # cumulative_fire = [sum(fire_levels_list[:i + 1]) for i in range(len(fire_levels_list))]
    # plt.figure(figsize=(10, 4))
    # plt.plot(iterations, cumulative_fire, color='orange')
    # plt.title('Cumulative Fire Levels')
    # plt.xlabel('Time step')
    # plt.ylabel('Cumulative Levels')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(OUTPUT_DIR / 'cumulative_fire_levels_plot.png')
    # plt.close()

    # # 5. 累积拥堵等级图
    # cumulative_congestion = [sum(congestion_levels_list[:i + 1]) for i in range(len(congestion_levels_list))]
    # plt.figure(figsize=(10, 4))
    # plt.plot(iterations, cumulative_congestion, color='purple')
    # plt.title('Cumulative Congestion Levels')
    # plt.xlabel('Time step')
    # plt.ylabel('Cumulative Levels')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(OUTPUT_DIR / 'cumulative_congestion_levels_plot.png')
    # plt.close()

    # # 6. 新增：总人数变化图
    # plt.figure(figsize=(10, 4))
    # plt.plot(iterations, total_people_list, color='#6a0dad', linestyle='-', marker='o', markersize=4)
    # plt.title('Total People in Evacuation Area')
    # plt.xlabel('Time step')
    # plt.ylabel('Number of People')
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(OUTPUT_DIR / 'total_people_plot.png', dpi=300)
    # plt.close()

    # print("六张分析图已保存：")
    # print("- static_field_plot.png        # 静态场趋势")
    # print("- fire_levels_plot.png         # 实时火灾等级")
    # print("- congestion_levels_plot.png   # 实时拥堵等级")
    # print("- cumulative_fire_levels_plot.png    # 累积火灾影响")
    # print("- cumulative_congestion_levels_plot.png  # 累积拥堵影响")
    # print("- total_people_plot.png        # 总人数变化\n")

    # # 生成动态图
    # create_gif(density_image_files, str(OUTPUT_DIR / 'people_number.gif'))
    # create_gif(fire_levels_image_files, str(OUTPUT_DIR / 'fire_evolution.gif'))
    # create_gif(congestion_image_files, str(OUTPUT_DIR / 'congestion_evolution.gif'))
    # create_gif(static_image_files, str(OUTPUT_DIR / 'static_field_evolution.gif'))

    # print("动态图生成完成：")
    # print("- density_evolution.gif       # 密度变化动态图")
    # print("- fire_evolution.gif          # 火灾发展动态图")
    # print("- congestion_evolution.gif    # 拥堵演变动态图")
    # print("- static_field_evolution.gif  # 静态场更新过程\n")
    # print("全部数据可视化完成！")

    # # 定义输出的文件名
    # output_file = OUTPUT_DIR / 'people_list.txt'

    # # 将列表写入txt文件
    # with output_file.open('w', encoding='utf-8') as file:
    #     for person in total_people_list:
    #         file.write(str(person) + "\n")  # 使用 str() 确保所有类型都能写入
    # print(rewards)
    # print(f"列表已成功写入到 {output_file}")
    # python -m utils.CTM.ctm_start.__init__
