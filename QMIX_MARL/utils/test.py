# import csv
#
# import networkx as nx
#
# def find_shortest_paths_to_exits(adj_matrix, exits):
#     G = nx.Graph()
#
#     # Add nodes and edges based on adjacency matrix
#     num_nodes = len(adj_matrix)
#     for i in range(num_nodes):
#         for j in range(i + 1, num_nodes):
#             if adj_matrix[i][j] != 0:
#                 G.add_edge(i + 1, j + 1, weight=int(1))  # Convert weight to int
#
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
#
#     return shortest_paths
#
# def change_door(original_door_size,no1, no2,number):
#     original_door_size[no1-1][no2-1] = number
#     original_door_size[no2-1][no1-1] = number
#
# def change_door_1(original_door_size,no1, no2,number):
#     original_door_size[no1][no2] = number
#     original_door_size[no2][no1] = number
#
# # Read CSV file
# original_door_size = []
# with open('../data_preprocessing/outputL.csv', 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         original_door_size.append([int(float(cell)) for cell in row])  # Convert each cell to integer
#
#
# # Manually modify adjacency matrix values
# node_203_index = 203 - 1
# node_204_index = 204 - 1
# node_199_index = 199 - 1
# node_200_index = 200 - 1
# node_184_index = 184 - 1
# node_185_index = 185 - 1
#
#
# original_door_size[node_203_index][node_204_index] = 3.45
# original_door_size[node_204_index][node_203_index] = 3.45
# original_door_size[node_199_index][node_200_index] = 3.45
# original_door_size[node_200_index][node_199_index] = 3.45
# original_door_size[node_184_index][node_185_index] = 2
# original_door_size[node_185_index][node_184_index] = 2
#
# change_door(original_door_size,164,165,3.1)
# change_door(original_door_size,165,166,3.1)
# change_door(original_door_size,167,166,3.1)
# change_door(original_door_size,167,109,3.1)
# change_door(original_door_size,198,164,3.1)
#
#
# change_door(original_door_size,177,178,2)
# change_door(original_door_size,178,179,2)
# change_door(original_door_size,179,180,2)
# change_door(original_door_size,180,187,2)
#
#
#
# change_door(original_door_size,181,182,2)
# change_door(original_door_size,182,183,2)
# change_door(original_door_size,183,184,2)
# change_door(original_door_size,184,185,2)
#
#
# change_door(original_door_size,199,200,3.45)
# change_door(original_door_size,200,201,3.45)
# change_door(original_door_size,201,202,3.45)
# change_door(original_door_size,202,203,3.45)
# change_door(original_door_size,203,204,3.45)
#
#
# change_door(original_door_size,205,168,1.6)
# change_door(original_door_size,168,169,1.6)
# change_door(original_door_size,169,170,1.6)
# change_door(original_door_size,217,172,1.6)
# change_door(original_door_size,172,171,1.6)
# change_door(original_door_size,171,170,1.6)
# change_door(original_door_size,170,120,4)
#
#
#
# change_door(original_door_size,228,173,2)
# change_door(original_door_size,173,174,2)
# change_door(original_door_size,174,175,2)
# change_door(original_door_size,175,176,2)
# change_door(original_door_size,176,147,2)
#
#
# change_door(original_door_size,71,117,2.6)
# change_door(original_door_size,109,63,3.25)
# change_door(original_door_size,157,48,3.25)
# change_door(original_door_size,134,97,3.25)
# change_door(original_door_size,138,96,3.25)
# change_door(original_door_size,159,73,4)
#
#
# change_door(original_door_size,214,215,3)
# change_door(original_door_size,213,214,3)
#
# change_door_1(original_door_size,48,157,0)
# change_door_1(original_door_size,63,109,0)
# change_door_1(original_door_size,71,117,0)
# change_door_1(original_door_size,73,159,0)
# change_door_1(original_door_size,96,138,0)
# change_door_1(original_door_size,97,137,0)
# change_door_1(original_door_size,101,102,0)
# change_door_1(original_door_size,109,167,0)
# change_door_1(original_door_size,120,170,0)
# change_door_1(original_door_size,147,176,0)
# change_door_1(original_door_size,168,205,0)
# change_door_1(original_door_size,170,120,0)
# change_door_1(original_door_size,172,217,0)
# change_door_1(original_door_size,173,228,0)
# change_door_1(original_door_size,174,146,0)
# change_door_1(original_door_size,198,164,0)
#
#
# exits = [100, 101, 102, 103]
# shortest_paths = find_shortest_paths_to_exits(original_door_size, exits)
#
# # Output the shortest paths in order of node numbers
# for node in sorted(shortest_paths.keys()):
#     print(f"Node {node}: Shortest path to nearest exit is {shortest_paths[node]} steps")
#
#
#



def find_connected_components(adj_matrix):
    def dfs(node, visited, component):
        stack = [node]
        while stack:
            current = stack.pop()
            if not visited[current]:
                visited[current] = True
                component.append(current)
                for neighbor in range(len(adj_matrix)):
                    if adj_matrix[current][neighbor] == 1 and not visited[neighbor]:
                        stack.append(neighbor)

    num_nodes = len(adj_matrix)
    visited = [False] * num_nodes
    group_list = []

    for node in range(num_nodes):
        if not visited[node]:
            component = []
            dfs(node, visited, component)
            group_list.append(component)

    return group_list

# 示例用法
adj_matrix = [
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]

group_list = find_connected_components(adj_matrix)
print(group_list)  # 输出: [[0, 1, 2, 3]]




