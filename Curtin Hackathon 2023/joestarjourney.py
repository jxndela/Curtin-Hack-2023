import pandas as pd
import math
from collections import defaultdict

# Load the data
planets_df = pd.read_csv("planets.csv")
refuel_locations_df = pd.read_csv("refuelLocations.csv")
wormholes_df = pd.read_csv("wormholes.csv")

class GraphWithCosts:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}
        self.refuel_costs = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance
        self.distances[(to_node, from_node)] = distance

    def add_refuel_cost(self, node, cost):
        self.refuel_costs[node] = cost

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def add_edges_based_on_proximity(graph, threshold=1000):
    for node1 in graph.nodes:
        for node2 in graph.nodes:
            if node1 != node2:
                distance = euclidean_distance(*node1, *node2)
                if distance <= threshold:
                    graph.add_edge(node1, node2, distance)
    return graph

def dijkstra_with_costs_and_supply(graph, initial, end, starting_supplies):
    rate_of_supply_depletion = 0.1

    unvisited_nodes = graph.nodes.copy()
    shortest_distance = {vertex: float('infinity') for vertex in graph.nodes}
    previous_nodes = {vertex: None for vertex in graph.nodes}
    shortest_distance[initial] = 0
    supplies = {vertex: starting_supplies if vertex == initial else 0 for vertex in graph.nodes}
    total_cost = {vertex: 0 for vertex in graph.nodes}
    
    while unvisited_nodes:
        current_node = min(unvisited_nodes, key=lambda vertex: (shortest_distance[vertex], -supplies[vertex]))
        unvisited_nodes.remove(current_node)
        
        if current_node == end or shortest_distance[current_node] == float('infinity'):
            break
        
        for neighbor in graph.edges[current_node]:
            path_distance = graph.distances[(current_node, neighbor)]
            supplies_used = path_distance * rate_of_supply_depletion
            remaining_supplies = supplies[current_node] - supplies_used
            
            if remaining_supplies < 0:
                if neighbor in graph.refuel_costs:
                    remaining_supplies = starting_supplies
                    total_cost[neighbor] = total_cost[current_node] + graph.refuel_costs[neighbor]
                else:
                    continue

            new_distance = shortest_distance[current_node] + path_distance
            
            if new_distance < shortest_distance[neighbor] or (new_distance == shortest_distance[neighbor] and remaining_supplies > supplies[neighbor]):
                shortest_distance[neighbor] = new_distance
                previous_nodes[neighbor] = current_node
                supplies[neighbor] = remaining_supplies

    path = []
    current_node = end
    while previous_nodes[current_node] is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    path.insert(0, current_node)

    if shortest_distance[end] == float('infinity'):
        return None, float('infinity'), float('infinity'), "Route failed due to insufficient starting supplies or no viable path."
    return path, shortest_distance[end], total_cost[end], None

# Construct the graph
graph_with_costs = GraphWithCosts()

for _, row in planets_df.iterrows():
    graph_with_costs.add_node((row['X'], row['Y']))
for _, row in refuel_locations_df.iterrows():
    node = (row['X'], row['Y'])
    graph_with_costs.add_node(node)
    graph_with_costs.add_refuel_cost(node, row['SupplyCostPer5LightYears'])

graph_with_costs = add_edges_based_on_proximity(graph_with_costs, threshold=1000)

for _, row in wormholes_df.iterrows():
    start = (row['StartX'], row['StartY'])
    end = (row['EndX'], row['EndY'])
    graph_with_costs.add_edge(start, end, 0)

earth_coords = tuple(planets_df[planets_df['Planet'] == 'Earth'][['X', 'Y']].values[0])
gliese_coords = tuple(planets_df[planets_df['Planet'] == 'Gliese'][['X', 'Y']].values[0])

# Example with 10,000 as starting supplies
shortest_path_with_supplies, distance_with_supplies, total_cost_with_supplies, failure_reason = dijkstra_with_costs_and_supply(graph_with_costs, earth_coords, gliese_coords, 10000)

print(f"Shortest Path: {shortest_path_with_supplies}")
print(f"Distance: {distance_with_supplies}")
print(f"Total Cost: {total_cost_with_supplies}")
print(f"Reason (if failed): {failure_reason}")
import matplotlib.pyplot as plt

def plot_graph_with_wormhole_check(graph, shortest_path=None):
    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot nodes
    for node in graph.nodes:
        if node in [earth_coords, gliese_coords]:
            ax.scatter(*node, color='red', s=100, zorder=5)  # Planets are in red
            if node == earth_coords:
                ax.text(node[0], node[1], 'Earth', fontsize=9, ha='right')
            else:
                ax.text(node[0], node[1], 'Gliese', fontsize=9, ha='right')
        else:
            ax.scatter(*node, color='blue', s=50, zorder=5)  # Refueling stations are in blue
    
    # Plot edges and check if wormhole is used in the shortest path
    wormhole_used = False
    for (node1, node2), distance in graph.distances.items():
        if distance == 0:  # Wormholes
            color = 'purple'
            linestyle = '--'
            if set([node1, node2]).issubset(shortest_path):  # Check if wormhole is in the shortest path
                color = 'magenta'
                linestyle = '-'
                wormhole_used = True
            ax.plot([node1[0], node2[0]], [node1[1], node2[1]], color=color, linestyle=linestyle, zorder=1)
        else:
            ax.plot([node1[0], node2[0]], [node1[1], node2[1]], color='grey', linestyle='-', linewidth=0.5, zorder=1)

    # Plot the shortest path
    if shortest_path:
        for i in range(len(shortest_path) - 1):
            start, end = shortest_path[i], shortest_path[i+1]
            ax.plot([start[0], end[0]], [start[1], end[1]], color='green', linewidth=2, zorder=2)
    
    ax.set_title("Visualization of Planets, Refueling Stations, Wormholes, and Shortest Path")
    plt.show()

    # Print if wormhole was used
    if wormhole_used:
        print("The shortest path uses a wormhole!")

# Plot the graph with the shortest path highlighted and check for wormhole usage
plot_graph_with_wormhole_check(graph_with_costs, shortest_path_with_supplies)