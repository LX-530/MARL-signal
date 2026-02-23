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
from collections import deque

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



free_velocity = 1.2  # 自由流速度
l = 4
max_density = 6  # 最大密度值 单位：人/米方
cri_density = 2.5  # 临界密度值 单位：人/米方
duration = l / free_velocity  # 时间步长，单位：秒
# Layer targets (requested split: layer4 -> layer3 -> layer2 -> layer1 exits).
LAYER2_TARGETS = [109, 117, 134, 138, 157]
LAYER3_TARGETS = [167, 170, 176]
LAYER4_TARGETS = [198, 205, 217, 228]

# Cross-layer one-way connectors (upstream -> downstream).
CONNECTOR_EDGES = [
    (198, 164), (205, 168), (217, 172), (228, 173),  # layer4 -> layer3
    (167, 109), (170, 120), (176, 147),              # layer3 -> layer2
    (109, 63), (117, 71), (134, 97), (138, 96), (157, 48),  # layer2 -> layer1
]


def _build_layer_sets(num_nodes):
    # Keep the exact user-requested ranges.
    return {
        "layer1": set(range(1, 105)),
        "layer2": set(range(104, 163)),
        "layer3": set(range(164, 177)),
        "layer4": set(range(177, num_nodes + 1)),
    }


# def find_shortest_paths_to_exits(adj_matrix, exits):
#     G = nx.Graph()

#     # Add nodes and edges based on adjacency matrix
#     num_nodes = len(adj_matrix)
#     for i in range(num_nodes):
#         for j in range(i + 1, num_nodes):
#             if adj_matrix[i][j] != 0:
#                 G.add_edge(i + 1, j + 1, weight=adj_matrix[i][j])

#     # Calculate shortest paths to all exit nodes
#     shortest_paths = {}
#     for node in G.nodes():
#         shortest_path_length = float('inf')
#         for exit_node in exits:
#             try:
#                 path_length = nx.shortest_path_length(G, source=node, target=exit_node, weight='weight')
#                 if path_length < shortest_path_length:
#                     shortest_path_length = path_length
#             except nx.NetworkXNoPath:
#                 continue
#         shortest_paths[node] = shortest_path_length

#     return shortest_paths, G

def visualize_graph(G, shortest_paths):
    pos = nx.spring_layout(G)  # positions for all nodes - seed for reproducibility

    labels = {node: f"{node} ({shortest_paths[node]})" for node in G.nodes()}

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=700, alpha=0.9, node_color='skyblue', ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5, edge_color='gray', ax=ax)
    nx.draw_networkx_labels(G, pos, labels, font_size=14, font_family="sans-serif", ax=ax)

    ax.set_title("Graph Visualization with Shortest Paths to Exits")
    plt.axis('off')  # Turn off the axis
    plt.show()

def _shortest_paths_layer(adj_matrix, layer_nodes, target_nodes):
    """Compute shortest path length (hop count) within a layer subgraph."""
    layer_nodes = sorted(layer_nodes)
    layer_set = set(layer_nodes)
    targets = [t for t in target_nodes if t in layer_set]

    G = nx.Graph()
    G.add_nodes_from(layer_nodes)

    for i in layer_nodes:
        row = adj_matrix[i - 1]
        for j in layer_nodes:
            if j > i and row[j - 1] != 0:
                G.add_edge(i, j, weight=1)

    if not targets:
        return {n: float('inf') for n in layer_nodes}

    lengths = nx.multi_source_dijkstra_path_length(G, targets, weight='weight')
    return {n: lengths.get(n, float('inf')) for n in layer_nodes}


def _shortest_paths_global(adj_matrix, target_nodes):
    """Fallback shortest path length (hop count) on full graph."""
    G = nx.Graph()
    num_nodes = len(adj_matrix)
    G.add_nodes_from(range(1, num_nodes + 1))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_matrix[i][j] != 0:
                G.add_edge(i + 1, j + 1, weight=1)

    targets = [t for t in target_nodes if 1 <= t <= num_nodes]
    if not targets:
        return {n: float('inf') for n in range(1, num_nodes + 1)}

    lengths = nx.multi_source_dijkstra_path_length(G, targets, weight='weight')
    return {n: lengths.get(n, float('inf')) for n in range(1, num_nodes + 1)}


def find_shortest_paths_to_exits(adj_matrix, exits):
    """
    Layer-aware shortest paths:
    layer4 -> layer3 targets, layer3 -> layer2 targets,
    layer2 -> layer1 targets, layer1 -> real exits.
    """
    num_nodes = len(adj_matrix)
    layers = _build_layer_sets(num_nodes)
    layer1 = layers["layer1"]
    layer2 = layers["layer2"]
    layer3 = layers["layer3"]
    layer4 = layers["layer4"]

    shortest_paths = {}
    shortest_paths.update(_shortest_paths_layer(adj_matrix, layer1, [e for e in exits if e in layer1]))
    shortest_paths.update(_shortest_paths_layer(adj_matrix, layer2, LAYER2_TARGETS))
    shortest_paths.update(_shortest_paths_layer(adj_matrix, layer3, LAYER3_TARGETS))
    shortest_paths.update(_shortest_paths_layer(adj_matrix, layer4, LAYER4_TARGETS))

    # The requested ranges skip node 163; fill any uncovered nodes via global fallback.
    all_nodes = set(range(1, num_nodes + 1))
    uncovered_nodes = all_nodes - set(shortest_paths.keys())
    if uncovered_nodes:
        fallback = _shortest_paths_global(adj_matrix, exits)
        for node_id in uncovered_nodes:
            shortest_paths[node_id] = fallback.get(node_id, float('inf'))

    return shortest_paths


def _build_shortest_path_edges(adj_matrix, layer_nodes, target_nodes, blocked_nodes=None):
    """
    Build a directed edge list that routes each node in a layer to the nearest target
    using hop-count shortest paths. Nodes in blocked_nodes are skipped (e.g., signal-controlled).
    """
    layer_nodes = sorted(layer_nodes)
    layer_set = set(layer_nodes)
    targets = [t for t in target_nodes if t in layer_set]
    blocked = set(blocked_nodes or [])

    if not targets:
        return []

    # BFS on the layer subgraph (unweighted)
    dist = {n: float('inf') for n in layer_nodes}
    dq = deque()
    for t in targets:
        dist[t] = 0
        dq.append(t)

    # Precompute neighbors within the layer
    neighbors = {}
    for i in layer_nodes:
        row = adj_matrix[i - 1]
        nbrs = []
        for j in layer_nodes:
            if j != i and row[j - 1] != 0:
                nbrs.append(j)
        neighbors[i] = nbrs

    while dq:
        u = dq.popleft()
        for v in neighbors[u]:
            if dist[v] == float('inf'):
                dist[v] = dist[u] + 1
                dq.append(v)

    edges = []
    for n in layer_nodes:
        if n in blocked or n in targets:
            continue
        if dist[n] == float('inf'):
            continue
        candidates = [v for v in neighbors[n] if dist.get(v, float('inf')) == dist[n] - 1]
        if not candidates:
            continue
        next_node = min(candidates)
        edges.append((n, next_node))
    return edges

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

    sigal_effect_cells = [#{1: [(68, 67), (1, 2), (2, 3), (3, 4),(67, 66), (66, 65)], 2: [(68, 1), (2, 1), (3, 2), (1, 100), (66, 67), (67, 2)], 4: [(3, 2), (2, 1), (1, 68), (68, 69),(66, 67), (67, 68)]},   # agent1
                          {1: [(69, 70), (70, 71), (71, 117), (117, 116), (116, 115),(115, 114), (104, 105), (105, 106),(106, 107)], 2: [(115, 116), (116, 117), (117, 71), (71, 70),(70, 69), (69, 68),(106,105),(105,104),(104,117)]},   # agent2
                          #{1: [(4, 5), (5, 6), (6, 7),(7, 8), (8, 9), (9, 10),(65, 64), (64, 63), (63, 62),(62, 61), (61, 60), (60, 59),], 3: [(9, 8), (8, 7), (7, 6),(6, 5), (5, 4), (4, 3),(60, 61), (61, 62), (62, 63),(63, 64), (64, 65), (65, 66)]},  # agent3
                          #{2: [(109, 63)], 4: [(109, 108)]},  # agent4
                          {1: [(114, 113), (113, 112),(112, 118), (111, 112),(110, 111), (108, 113),(107, 114)], 2: [(112, 111), (111, 110),(110, 109), (113, 108),(108, 109), (114, 107),(107, 108)], 3: [(112, 113), (113, 114),(114, 115), (110, 111),(111, 108), (108, 107),(107, 106)]},  # agent5
                          {1: [(118, 119), (119, 121),(121, 122), (122, 127),(120, 119),(124, 125), (123, 126)], 3: [(124, 123), (123, 122),(122, 121), (121, 119),(120, 119),(119, 118), (118, 112)]},  # agent6
                          #{1: [(10, 11), (11, 12),(12, 13), (13, 14),(14, 15), (15, 16),(16, 17), (17, 18),(59, 58), (58, 57),(57, 56), (56, 55),(55, 54), (54, 53),(53, 52), (52, 51)], 3: [(17, 16), (16, 15),(15, 14), (14, 13),(13, 12), (12, 11),(11, 10), (10, 9),(52, 53), (53, 54),(54, 55), (55, 56),(56, 57), (57, 58),(58, 59), (59, 60)]},  # agent7
                          {1: [(125, 130), (130, 131), (131, 136),(136, 135), (126, 129), (129, 132),(132, 135), (135, 137), (137, 140),(140, 141), (141, 146), (127, 128),(128, 133), (133, 134), (134, 138),(138, 139), (139, 142), (142, 145)], 3: [(141, 140), (140, 137), (137, 135),(135, 132), (132, 129), (129, 126),(126, 123), (125, 126), (130, 125),(131, 130), (136, 131), (142, 139),(139, 138), (138, 134), (134, 133),(133, 128), (128, 127), (127, 122)], 4: [(125, 126), (126, 127), (127, 128),(130, 129), (129, 132), (131, 132),(132, 135), (136, 135), (128, 133),(133, 134), (135, 134), (134, 97),(141, 140), (140, 137), (137, 138),(138, 96), (142, 139), (139, 138)] },  # agent8
                         # {1: [(99, 90), (90, 89),(98, 91), (91, 88),(97, 92), (92, 87),(96, 93), (93, 86),(95, 94), (94, 85),(89, 88), (88, 87) , (87, 86),(86, 85), (85, 84),(84, 83), (83, 82),(82, 81)], 4: [(99, 90), (90, 89),(98, 91), (91, 88),(97, 92), (92, 87),(96, 93), (93, 86),(95, 94), (94, 85),(89, 88), (88, 87) , (87, 86),(86, 85), (85, 101),(84, 85), (83, 84),(82, 83)]},  # agent9
                          {1: [(147, 148), (148, 157), (157, 158),(146, 149), (149, 156), (156, 159),(145, 150), (150, 155), (155, 160), (143, 144), (144, 151), (151, 154),(154, 161),(152, 153), (153, 154), (163, 162),(162, 161),(161, 160),(160, 159), (158, 159), (159, 73)],2: [(147, 148), (148, 157), (158, 157),(143, 144),(144, 151), (145, 150), (146, 149),(149, 156), (150, 155), (151, 154),(152, 153), (153, 154), (154, 155),(155, 156),(156, 157), (159, 158), (163, 162),(162, 161),(161, 160),(160, 159), (157, 48)], 3: [(147, 146), (148, 149), (157, 156),(158, 159), (159, 156), (156, 149), (149, 146),(146, 141), (160, 155), (155, 150),(150, 145), (145, 142), (161, 154), (154, 151),(151, 144),(144, 143), (143, 142),(152, 153), (153, 154),(163, 162),(162, 161)]},  # agent10
                          #{1: [(18, 19), (19, 20), (20, 39),(39, 40), (40, 41), (41, 42),(42, 37), (43, 38), (51, 50),(50, 49), (49, 48), (48, 47),(47, 46), (46, 45), (45, 42),(44, 43) ], 3: [ (43, 42), (42, 41), (41, 40),(40, 39), (39, 20), (20, 19),(19, 18), (18, 17), (44, 45),(45, 46), (46, 47), (47, 48),(48, 49), (49, 50), (50, 51),(51, 52) ], 4: [ (18, 19), (19, 20), (20, 39),(39, 40), (40, 41), (43, 42),(42, 41), (44, 45), (45, 46),(51, 50), (50, 49), (49, 48),(48, 47), (47, 46), (41, 46),(46, 72) ]},  # agent11
                          #{2: [(23, 24), (24, 25), (25, 26), (26, 27), (22, 33), (33, 32),(32, 31), (31, 30), (21, 34),(34, 35), (35, 36), (36, 37),(37, 30), (30, 27), (27, 102),(38, 29), (29, 28), (28, 103)], 4: [(23, 22), (22, 21), (21, 20), (24, 33), (33, 34), (34, 39),(25, 32), (32, 35), (35, 40),(26, 31), (31, 36), (36, 41),(27, 30), (30, 37), (37, 42),(28, 29), (29, 38), (38, 43)]},  # agent12
                          #{4: [(72, 73),(73, 74),(74, 75),(75, 76)], 2: [(75, 74),(74, 73),(73, 72),(72, 46)]},  # agent13
                          #{2: [(76, 77), (77, 78), (78, 79),(79, 80), (80, 81), (81, 82)], 3: [(81, 80), (80, 79), (79, 78),(78, 77), (77, 76), (76, 75)]},  # agent14
                          {1: [(187, 188), (188, 193), (193, 194),(194, 199), (199, 200), (200, 201),(201, 202)], 4: [(187, 188), (188, 193), (193, 194), (201, 200), (200, 199), (199, 198),(194, 195)]},  # agent15
                          {1: [(185, 190), (190, 191), (191, 196),(196, 197), (197, 210), (210, 209),(209, 208)], 2: [(209, 210), (210, 197), (197, 198),(185, 190), (190, 191), (191, 196),(196, 195)]},  # agent16
                          {1: [(202, 203), (203, 204),(204, 223), (223, 222),(222, 221),(221, 220)], 3: [(221, 222), (222, 223),(223, 204), (204, 203), (203, 202),(202, 201)], 4: [(202, 203), (203, 204),(204, 205), (221, 222),(222, 223),(223, 204)]},  # agent17
                          {1: [(208, 207), (207, 206),(206, 211), (211, 212),(212, 213),(213, 214)], 2: [(208, 207), (207, 206),(213, 212),  (212, 211),(211, 206),(206, 205)], 3: [(213, 212), (212, 211),(211, 206), (206, 207),(207, 208),(208, 209)]},  # agent18
                          {1: [(220, 219), (219, 218),(218, 217), (224, 225),(229, 228)], 3: [(229, 224), (224, 218),(218, 219),  (219, 220),(220, 221)]},  # agent19
                          {1: [(217, 225), (225, 228), (228, 173)], 2: [(225, 228), (228, 173), (217, 172)], 3: [(225, 217), (228, 173), (217, 172)],4: [(225, 217), (228, 225), (217, 172)]},  # agent20
                          {1: [(214, 215), (215, 216), (216, 217), (226, 225), (227, 228)],3: [(227, 226), (226, 216), (216, 215), (215, 214), (214, 213)]},  # agent21
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
                  (27, 102), (38, 29), (29, 28), (28, 103),
                  (75, 74), (74, 73), (73, 72), (72, 46), (81, 80), (80, 79), (79, 78), (78, 77), (77, 76), (76, 75)

                  ]

    # agent_cell_ids = [32, 43, 46, 50, 51, 57, 60, 62, 18, 13, 11, 10, 8, 5, 25, 17, 30, 66] #15,

    agent_cell_ids = [71,113,119,134,156,194,196,204,206,224,225,226]#[1,71,6,109,113,119,14,134,85,156,39,31,73,78,194,196,204,206,224,225,226]
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

    exits = [100, 101, 102, 103]  #鍑哄彛鐨勭紪鍙?

    # (layered shortest-path fixed directions moved below)

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

    # ---- Layered shortest-path default directions ----
    num_nodes = len(original_door_size)
    layers = _build_layer_sets(num_nodes)
    layer1 = layers["layer1"]
    layer2 = layers["layer2"]
    layer3 = layers["layer3"]
    layer4 = layers["layer4"]

    connector_edges = list(CONNECTOR_EDGES)

    # Build shortest-path directions within each layer.
    fixed_cell = []
    fixed_cell += _build_shortest_path_edges(original_door_size, layer1, [e for e in exits if e in layer1], blocked_nodes=None)
    fixed_cell += _build_shortest_path_edges(original_door_size, layer2, LAYER2_TARGETS, blocked_nodes=None)
    fixed_cell += _build_shortest_path_edges(original_door_size, layer3, LAYER3_TARGETS, blocked_nodes=None)
    fixed_cell += _build_shortest_path_edges(original_door_size, layer4, LAYER4_TARGETS, blocked_nodes=None)
    fixed_cell += connector_edges

    # Manual edge overrides + force all cross-layer connectors one-way.
    manual_blocked_edges = {(135, 98), (98, 135), (159, 73), (73, 159)}
    reverse_connector_edges = {(b, a) for (a, b) in connector_edges}
    blocked_edges = manual_blocked_edges | reverse_connector_edges
    forced_edges = list(connector_edges)

    fixed_cell = [e for e in fixed_cell if e not in blocked_edges]
    for e in forced_edges:
        if e not in fixed_cell:
            fixed_cell.append(e)

    energy_domine = find_shortest_paths_to_exits(original_door_size, exits)


    Position_cells =[]

    with (DATA_DIR / 'outputP.csv').open('r') as file:
        reader = csv.reader(file)
        for row in reader:
            list1 = []
            for row1 in row:
                list1.append(eval(row1))
            Position_cells.append(list1)


    TF_exit = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]

    return baseGraph(
        TF_exit,
     Position_cells,
           people_number,
     original_door_size,
        sigal_effect_cells,
        fixed_cell,
     fire_info,
     agent_cell_ids,
     energy_domine,
    cell_type,
    blocked_edges,
    forced_edges
    )


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
            else:
                print(f"Warning: Connection {start}->{end} skipped (invalid node)")
    return colors  # 返回颜色用于图例

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


def _flatten_directions(nested_connections):
    if not nested_connections:
        return []
    edges = []
    for group in nested_connections:
        edges.extend(group)
    return edges


def compute_layers_from_directions(num_nodes, nested_connections, exit_nodes):
    """
    Layer definition:
      layer 0 = exit nodes
      layer 1 = one step upstream to exits
      layer k = k steps upstream
    """
    edges = _flatten_directions(nested_connections)
    if not edges or not exit_nodes:
        return {}

    rev_adj = [[] for _ in range(num_nodes)]
    for start, end in edges:
        rev_adj[end - 1].append(start - 1)

    dist = [None] * num_nodes
    dq = deque()
    for e in exit_nodes:
        if 1 <= e <= num_nodes:
            dist[e - 1] = 0
            dq.append(e - 1)

    while dq:
        u = dq.popleft()
        du = dist[u]
        for v in rev_adj[u]:
            if dist[v] is None:
                dist[v] = du + 1
                dq.append(v)

    return {i + 1: d for i, d in enumerate(dist) if d is not None}


def find_entrance_nodes(num_nodes, nested_connections, exit_nodes):
    """
    Entrance = source nodes in the directed graph (no incoming edges), excluding exits.
    """
    edges = _flatten_directions(nested_connections)
    if not edges:
        return []

    indeg = [0] * num_nodes
    outdeg = [0] * num_nodes
    for start, end in edges:
        outdeg[start - 1] += 1
        indeg[end - 1] += 1

    entrance_nodes = []
    exit_set = set(exit_nodes or [])
    for i in range(num_nodes):
        node_id = i + 1
        if node_id in exit_set:
            continue
        if outdeg[i] > 0 and indeg[i] == 0:
            entrance_nodes.append(node_id)
    return entrance_nodes


def visualize_layers(exit_nodes=None, step=1, nested_connections=None, max_layer=3, show_labels=True):
    """
    Visualize layer (distance-to-exit in directed graph) and mark entrances/exits.
    """
    csv_path = DATA_DIR / 'outputP.csv'

    def parse_coordinate(cell):
        matches = re.findall(r"[-+]?\d*\.?\d+", str(cell))
        return float(matches[0]), float(matches[1])

    df = pd.read_csv(csv_path, header=None, encoding="utf-8")
    num_nodes = len(df)

    first_poly = [parse_coordinate(df.iloc[0, col]) for col in range(4)]
    dx, dy = -min(p[0] for p in first_poly), -max(p[1] for p in first_poly)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_aspect("equal")
    ax.grid(False)

    centers = {}
    all_points = []

    layer_map = compute_layers_from_directions(num_nodes, nested_connections, exit_nodes or [])
    entrance_nodes = find_entrance_nodes(num_nodes, nested_connections, exit_nodes or [])

    layer_colors = {
        0: "#2ecc71",  # exit
        1: "#3498db",
        2: "#f39c12",
        3: "#9b59b6",
    }
    default_layer_color = "#95a5a6"
    unreachable_color = "#bdc3c7"

    for idx, row in df.iterrows():
        poly_points = []
        node_id = idx + 1
        layer = layer_map.get(node_id)

        if exit_nodes and node_id in exit_nodes:
            use_color = layer_colors[0]
            label_text = "E"
        elif layer is None:
            use_color = unreachable_color
            label_text = ""
        else:
            use_color = layer_colors.get(layer, default_layer_color if layer > max_layer else layer_colors.get(layer, default_layer_color))
            label_text = str(layer)

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
            alpha=0.75
        )
        ax.add_patch(poly)

        center = np.mean(poly_points, axis=0)
        centers[node_id] = center

        if show_labels and label_text:
            ax.text(*center, label_text, fontsize=7, color='black',
                    ha='center', va='center')

    # mark entrances on top of layer colors
    if entrance_nodes:
        for node_id in entrance_nodes:
            if node_id in centers:
                ax.scatter(centers[node_id][0], centers[node_id][1], s=20,
                           color="#2980b9", marker='o', zorder=3)

    # optional: draw directions
    if nested_connections:
        draw_nested_connections(ax, centers, nested_connections)

    all_points = np.array(all_points)
    padding = 0.005
    ax.set_xlim(all_points[:, 0].min() - padding, all_points[:, 0].max() + padding)
    ax.set_ylim(all_points[:, 1].min() - padding, all_points[:, 1].max() + padding)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_title(f"Layers at Step {step}", fontsize=14)

    legend_patches = [
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=layer_colors[0], markersize=10, label='Exit (Layer 0)'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor="#2980b9", markersize=6, label='Entrance (source)'),
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=layer_colors[1], markersize=10, label='Layer 1'),
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=layer_colors[2], markersize=10, label='Layer 2'),
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=layer_colors[3], markersize=10, label='Layer 3'),
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=unreachable_color, markersize=10, label='Unreachable'),
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize='x-small')

    output_filename = OUTPUT_DIR / f"layers_step{step}.png"
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
    for i in range(0, 97):
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
        density_image_files.append(visualize_numbers(g.agent_cell_ids, [100, 101, 102, 103],
                                                     g.current_num, i, g.groups_directions, 'people'))
        fire_levels_image_files.append(visualize_numbers(g.agent_cell_ids, [100, 101, 102, 103],
                                                         g.fire_levels, i, g.groups_directions, 'fire_levels'))
        congestion_image_files.append(visualize_numbers(g.agent_cell_ids, [100, 101, 102, 103],
                                                        g.congestion_levels, i, g.groups_directions, 'congestion'))
        static_image_files.append(visualize_numbers(g.agent_cell_ids, [100, 101, 102, 103],
                                                    g.static_field, i, g.groups_directions, 'static'))
        # visualize_numbers(g.agent_cell_ids, [100, 101, 102, 103],
        #                                              location, i, g.groups_directions, 'location')
    #
    # 1. 静态场变化图
    plt.figure(figsize=(10, 4))
    plt.plot(iterations, static_fields, color='blue')
    plt.title('Static Field over Iterations')
    plt.xlabel('Time step')
    plt.ylabel('Static Field')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'static_field_plot.png')
    plt.close()

    # 2. 火灾等级变化图
    plt.figure(figsize=(10, 4))
    plt.plot(iterations, fire_levels_list, color='red')
    plt.title('Fire Levels over Iterations')
    plt.xlabel('Time step')
    plt.ylabel('Fire Levels')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fire_levels_plot.png')
    plt.close()

    # 3. 拥堵等级变化图
    plt.figure(figsize=(10, 4))
    plt.plot(iterations, congestion_levels_list, color='green')
    plt.title('Congestion Levels over Iterations')
    plt.xlabel('Time step')
    plt.ylabel('Congestion Levels')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'congestion_levels_plot.png')
    plt.close()

    # 4. 累积火灾等级图
    cumulative_fire = [sum(fire_levels_list[:i + 1]) for i in range(len(fire_levels_list))]
    plt.figure(figsize=(10, 4))
    plt.plot(iterations, cumulative_fire, color='orange')
    plt.title('Cumulative Fire Levels')
    plt.xlabel('Time step')
    plt.ylabel('Cumulative Levels')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cumulative_fire_levels_plot.png')
    plt.close()

    # 5. 累积拥堵等级图
    cumulative_congestion = [sum(congestion_levels_list[:i + 1]) for i in range(len(congestion_levels_list))]
    plt.figure(figsize=(10, 4))
    plt.plot(iterations, cumulative_congestion, color='purple')
    plt.title('Cumulative Congestion Levels')
    plt.xlabel('Time step')
    plt.ylabel('Cumulative Levels')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cumulative_congestion_levels_plot.png')
    plt.close()

    # 6. 新增：总人数变化图
    plt.figure(figsize=(10, 4))
    plt.plot(iterations, total_people_list, color='#6a0dad', linestyle='-', marker='o', markersize=4)
    plt.title('Total People in Evacuation Area')
    plt.xlabel('Time step')
    plt.ylabel('Number of People')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'total_people_plot.png', dpi=300)
    plt.close()

    print("六张分析图已保存：")
    print("- static_field_plot.png        # 静态场趋势")
    print("- fire_levels_plot.png         # 实时火灾等级")
    print("- congestion_levels_plot.png   # 实时拥堵等级")
    print("- cumulative_fire_levels_plot.png    # 累积火灾影响")
    print("- cumulative_congestion_levels_plot.png  # 累积拥堵影响")
    print("- total_people_plot.png        # 总人数变化\n")

    # 生成动态图
    create_gif(density_image_files, str(OUTPUT_DIR / 'people_number.gif'))
    create_gif(fire_levels_image_files, str(OUTPUT_DIR / 'fire_evolution.gif'))
    create_gif(congestion_image_files, str(OUTPUT_DIR / 'congestion_evolution.gif'))
    create_gif(static_image_files, str(OUTPUT_DIR / 'static_field_evolution.gif'))

    print("动态图生成完成：")
    print("- density_evolution.gif       # 密度变化动态图")
    print("- fire_evolution.gif          # 火灾发展动态图")
    print("- congestion_evolution.gif    # 拥堵演变动态图")
    print("- static_field_evolution.gif  # 静态场更新过程\n")
    print("全部数据可视化完成！")

    # 定义输出的文件名
    output_file = OUTPUT_DIR / 'people_list.txt'

    # 将列表写入txt文件
    with output_file.open('w', encoding='utf-8') as file:
        for person in total_people_list:
            file.write(str(person) + "\n")  # 使用 str() 确保所有类型都能写入
    print(rewards)
    print(f"列表已成功写入到 {output_file}")
#     cd D:\github\signal\MARL-signal\QMIX_MARL
# New-Item -ItemType Directory -Force .\.mplconfig | Out-Null
# $env:PYTHONPATH="$PWD"
# $env:MPLCONFIGDIR="$PWD\.mplconfig"
# python -m utils.CTM.ctm_start.__init__
