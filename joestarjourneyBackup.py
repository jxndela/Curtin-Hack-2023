import math
import csv
import os
import matplotlib.pyplot as plt

class SpaceGraph:
    def __init__(self):
        self.edges = []
        self.metadata = {}
        self.nodes = set()  # Adding a set to store 

    def add_edge(self, node1, node2, weight):
        # Store the edge
        self.edges.append((node1, node2, weight))

        # If node1 is a dictionary, extract coordinates and store metadata
        if isinstance(node1, dict):
            x, y = float(node1['X']), float(node1['Y'])
            self.metadata[(x, y)] = node1

        # Similarly, if node2 is a dictionary
        if isinstance(node2, dict):
            x, y = float(node2['X']), float(node2['Y'])
            self.metadata[(x, y)] = node2

        self.nodes.add(get_node_key(node1))
        self.nodes.add(get_node_key(node2))

    def get_metadata(self, node):
        return self.metadata.get(node, None)
    
def process_data(graph, data):
    # Here, data might be your problematic edges or any other source
    for item in data:
        start, end, weight = item
        graph.add_edge(start, end, weight)

# Load data from CSV
def load_csv(filename):
    with open(filename, newline='') as csvfile:
        return list(csv.DictReader(csvfile))

# Read source and destination from planets.csv
def read_planet_coordinates(planet_name):
    with open('planets.csv', 'r') as csvfile:
        planet_reader = csv.DictReader(csvfile)
        for row in planet_reader:
            if row["Planet"] == planet_name:
                return (float(row["X"]), float(row["Y"]))

# Calculate Euclidean distance
def calculateEuclideanDistance(point1, point2):
    x1, y1 = float(point1[0]), float(point1[1])
    x2, y2 = float(point2[0]), float(point2[1])
    
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# Create nodes as tuples for consistency
def get_node_key(node):
    if isinstance(node, dict):
        return (float(node["X"]), float(node["Y"]))
    elif isinstance(node, tuple):
        return node
    else:
        raise TypeError("Unexpected type for node")

def createGraph(planets, refuelLocations, wormholes):
    graph = SpaceGraph()

    # Connect each planet to every other planet and refuel location
    for planet in planets:
        for other_planet in planets:
            if planet != other_planet:
                graph.add_edge(
                    planet,
                    other_planet,
                    calculateEuclideanDistance((planet["X"], planet["Y"]), (other_planet["X"], other_planet["Y"]))
                )
        for refuel in refuelLocations:
            graph.add_edge(
                planet,
                refuel,
                calculateEuclideanDistance((planet["X"], planet["Y"]), (refuel["X"], refuel["Y"]))
            )

    # Connect each refuel location to every other refuel location and every planet
    for refuel in refuelLocations:
        for other_refuel in refuelLocations:
            if refuel != other_refuel:
                graph.add_edge(
                    refuel,
                    other_refuel,
                    calculateEuclideanDistance((refuel["X"], refuel["Y"]), (other_refuel["X"], other_refuel["Y"]))
                )
        for planet in planets:
            graph.add_edge(
                refuel,
                planet,
                calculateEuclideanDistance((refuel["X"], refuel["Y"]), (planet["X"], planet["Y"]))
            )

    # Connect each planet and refuel location to every wormhole start and end
    for wormhole in wormholes:
        start_coords = (wormhole["StartX"], wormhole["StartY"])
        end_coords = (wormhole["EndX"], wormhole["EndY"])

        for planet in planets:
            graph.add_edge(
                planet,
                start_coords,
                calculateEuclideanDistance((planet["X"], planet["Y"]), start_coords)
            )
            graph.add_edge(
                planet,
                end_coords,
                calculateEuclideanDistance((planet["X"], planet["Y"]), end_coords)
            )

        for refuel in refuelLocations:
            graph.add_edge(
                refuel,
                start_coords,
                calculateEuclideanDistance((refuel["X"], refuel["Y"]), start_coords)
            )
            graph.add_edge(
                refuel,
                end_coords,
                calculateEuclideanDistance((refuel["X"], refuel["Y"]), end_coords)
            )

    # Add wormholes as edges with a distance of 0
    for wormhole in wormholes:
        start_coords = (wormhole["StartX"], wormhole["StartY"])
        end_coords = (wormhole["EndX"], wormhole["EndY"])

        start_data = graph.get_metadata(start_coords) or {'X': wormhole["StartX"], 'Y': wormhole["StartY"], 'Name': 'Unknown'}
        end_data = graph.get_metadata(end_coords) or {'X': wormhole["EndX"], 'Y': wormhole["EndY"], 'Name': 'Unknown'}

        graph.add_edge(start_data, end_data, 0)
        
    return graph

def dijkstra(nodes, edges, start, end):
    # Initialize distances and predecessors
    distances = {node: float('infinity') for node in nodes}
    predecessors = {node: None for node in nodes}
    distances[start] = 0

    unvisited = nodes.copy()

    while unvisited:
        # Node with the smallest distance will be the first node in sorted_nodes
        current_node = min(unvisited, key=lambda node: distances[node])
        unvisited.remove(current_node)

        # Break if we've reached the end or if the smallest distance is infinity (disconnected graph)
        if distances[current_node] == float('infinity') or current_node == end:
            break

        # Find neighbors
        neighbors = [edge[1] for edge in edges if edge[0] == current_node and edge[1] in unvisited]

        for neighbor in neighbors:
            # Extract the distance from the edge data
            edge_distance = next(edge[2] for edge in edges if edge[0] == current_node and edge[1] == neighbor)
            
            new_distance = distances[current_node] + edge_distance
            
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                predecessors[neighbor] = current_node

    # Reconstruct the path from end to start by following predecessors
    path = []
    while end:
        path.insert(0, end)
        end = predecessors[end]

    # Construct edges_in_path based on the path list
    edges_in_path = [(path[i], path[i+1]) for i in range(len(path) - 1)]

    # Get the total distance from the start to the end
    distance = distances[path[-1]]

    return path, edges_in_path, distance

def plot_full_graph(nodes, edges, edges_in_path, wormholes):
    plt.figure(figsize=(10, 10))
    
    # Extract wormhole entrance and exit points
    wormhole_points = []
    for wormhole in wormholes:
        start = get_node_key((float(wormhole['StartX']), float(wormhole['StartY'])))
        end = get_node_key((float(wormhole['EndX']), float(wormhole['EndY'])))
        wormhole_points.append(start)
        wormhole_points.append(end)
    
    # Plot wormhole points as orange dots
    for point in wormhole_points:
        plt.scatter(point[0], point[1], c='orange')
    
    # Plot all other nodes as blue dots
    for node in nodes:
        if node not in wormhole_points:
            plt.scatter(node[0], node[1], c='blue')
    
    # Plot all edges (connections) as light gray lines with debugging info
    for edge in edges:
        try:
            x_values = [edge[0][0], edge[1][0]]
            y_values = [edge[0][1], edge[1][1]]
            plt.plot(x_values, y_values, 'k-', linewidth=0.5, alpha=0.3)  # Use a low alpha value to make it less opaque
        except Exception as e:
            print(f"DEBUGGING: Problematic edge -> {edge}. Error -> {e}")  # This will print out the problematic edge structure and the associated error

    
    # Plot wormholes with a specific style (green dashed)
    for wormhole in wormholes:
        start = get_node_key((float(wormhole['StartX']), float(wormhole['StartY'])))
        end = get_node_key((float(wormhole['EndX']), float(wormhole['EndY'])))
        plt.plot([start[0], end[0]], [start[1], end[1]], 'g--')
    
    # Plot the edges of the path in red
    for edge in edges_in_path:
        x_values = [edge[0][0], edge[1][0]]
        y_values = [edge[0][1], edge[1][1]]
        plt.plot(x_values, y_values, 'r-')
        
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Full Graph Visualization')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Check if the required files exist
    required_files = ["planets.csv", "refuelLocations.csv", "wormholes.csv"]
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: {file} not found.")
            exit(1)

    # Load data
    planets = load_csv("planets.csv")
    refuelLocations = load_csv("refuelLocations.csv")
    wormholes = load_csv("wormholes.csv")

    graph = createGraph(planets, refuelLocations, wormholes)
    nodes = list(graph.nodes)  # Convert the nodes set to a list
    edges = graph.edges


    
    # Allow user to specify source and destination
    source_name = input("Enter the source planet: ")
    destination_name = input("Enter the destination planet: ")

    source = read_planet_coordinates(source_name)
    destination = read_planet_coordinates(destination_name)

    if source is None or destination is None:
        print("Invalid source or destination!")
        exit(1)

    path, edges_in_path, distance = dijkstra(nodes, edges, source, destination)

    if not path:
        print(f"No path found from {source_name} to {destination_name}!")
        exit(1)

    # Initialize a variable to hold the cumulative weight
    cumulative_distance = 0.0

    # print out the final shortest path, distance, and cumulative weight
    print("\nShortest Path:")
    for edge in edges_in_path:
        distance = next(e[2] for e in edges if e[0] == edge[0] and e[1] == edge[1])
        cumulative_distance += distance
        
        if distance == 0:  # This is a wormhole
            print(f"From {edge[0]} to {edge[1]}: Used a wormhole!")
        else:
            print(f"From {edge[0]} to {edge[1]}: Distance = {distance}, Cumulative Distance = {cumulative_distance}")
    print("\nTotal Distance:", cumulative_distance)

    # At the end, after finding the path:
    plot_full_graph(nodes, edges, edges_in_path, wormholes)