import numpy as np
import matplotlib.pyplot as plt
import json

import sys
from collections import namedtuple
from itertools import permutations
from typing import Dict
from scipy.spatial import KDTree
import networkx as nx

import time
import pdb
import random

Node = namedtuple('Node', ['id', 'x', 'y'])
Edge = namedtuple('Edge', ['src', 'tgt'])



def on_segment(segment_start,point,segment_end):
    collinear = ((point[1] - segment_start[1]) * (segment_end[0] - segment_start[0]) ==
                 (segment_end[1] - segment_start[1]) * (point[0] - segment_start[0]))

    if not collinear:
        return False

    # Check if the point lies between the segment's endpoints
    min_x, max_x = sorted([segment_start[0], segment_end[0]])
    min_y, max_y = sorted([segment_start[1], segment_end[1]])

    # Check for strict inequality for both x and y coordinates
    return (min_x < point[0] < max_x or min_y < point[1] < max_y)

def overlap_vectorized(A1, A2, B1, B2):
    overlaps = np.zeros(len(A1), dtype=bool)

    for i, (a1, a2, b1, b2) in enumerate(zip(A1, A2, B1, B2)):

        if (on_segment(a1, b1, a2) or on_segment(a1, b2, a2) or
            on_segment(b1, a1, b2) or on_segment(b1, a2, b2)):
            overlaps[i] = True
    #pdb.set_trace()
    return overlaps


def edges_crossed(a1, a2, b1, b2):
    return np.cross(a2 - a1, b1 - a1) * np.cross(a2 - a1, b2 - a1) < 0, \
           np.cross(b2 - b1, a1 - b1) * np.cross(b2 - b1, a2 - b1) < 0

def count_crossings(nodes, edges):
    n_edges = edges.shape[0]
    edge_pairs = np.array([(i, j) for i in range(n_edges) for j in range(i, n_edges)])
    src1, tgt1 = edges[edge_pairs[:, 0], :2].astype(int).T
    src2, tgt2 = edges[edge_pairs[:, 1], :2].astype(int).T
    p1, q1 = nodes[src1], nodes[tgt1]
    p2, q2 = nodes[src2], nodes[tgt2]
    
    cond1, cond2 = edges_crossed(p1, q1, p2, q2)
    overlaps = overlap_vectorized(p1, q1, p2, q2)
    crossed = np.logical_and(cond1, cond2)
    crossed_counts = np.bincount(edge_pairs[:, 0], crossed) + np.bincount(edge_pairs[:, 1], crossed)
    overlaps_counts = np.bincount(edge_pairs[:, 0], overlaps) + np.bincount(edge_pairs[:, 1], overlaps)
    overlaps_counts = overlaps_counts>0
    #pdb.set_trace()
    edges[:, 2] = crossed_counts[:n_edges] + overlaps_counts[:n_edges]*len(edges)

    return edges

def fruchterman_reingold(nodes, edges, iterations=1000, k=None, width=1, height=1, seed=None):
    np.random.seed(seed)
    n_nodes = nodes.shape[0]
    pos = np.random.rand(n_nodes, 2) * np.array([width, height])
    forces = np.zeros((n_nodes, 2))

    if k is None:
        k = np.sqrt(width * height / n_nodes)

    def repulsive_force(d, k):
        return k * k / d

    def attractive_force(d, k):
        return d * d / k

    for i in range(iterations):
        print("iter=",i)
        # Compute repulsive forces
        forces.fill(0)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                delta = pos[i] - pos[j]
                distance = np.linalg.norm(delta)
                forces[i] += (delta / distance) * repulsive_force(distance, k)
                forces[j] -= (delta / distance) * repulsive_force(distance, k)

        # Compute attractive forces
        for i in range(edges.shape[0]):
            src = int(edges[i][0])
            tgt = int(edges[i][1])
            delta = pos[src] - pos[tgt]
            distance = np.linalg.norm(delta)
            forces[src] -= (delta / distance) * attractive_force(distance, k)
            forces[tgt] += (delta / distance) * attractive_force(distance, k)

        # Update positions
        pos += forces / np.linalg.norm(forces, axis=1)[:, np.newaxis]

    return pos

def store_output(points, nodes,edges,file_name):
    # Convert numpy arrays to dictionaries
    points_list = [{"x": int(x), "y": int(y)} for x, y in points]
    nodes_list = [{"id": i, "x": int(x), "y": int(y)} for i, (x, y) in enumerate(nodes)]
    edges_list = [{"source": int(src), "target": int(tgt)} for src, tgt in edges]
    #pdb.set_trace()
    # Store dictionaries in a single dictionary
    data = {"points": points_list, "nodes": nodes_list, "edges": edges_list}

    # Save data as a JSON file
    with open(file_name, "w") as outfile:
        json.dump(data, outfile, indent=4)


def nearest_neighbor(points, positions):
    # Find the unique nearest point for each node
    tree = KDTree(points)
    nearest_points = []
    used_points_indices = set()
    for node in positions:
        distances, point_indices = tree.query(node, k=len(points))

        # Iterate through the nearest points until we find an unassigned point
        for i in range(len(point_indices)):
            nearest_point_index = point_indices[i]
            if nearest_point_index not in used_points_indices:
                nearest_point = points[nearest_point_index]
                nearest_points.append(nearest_point)
                used_points_indices.add(nearest_point_index)
                break

    nearest_points = np.array(nearest_points)

    return nearest_points

def visualize(points, nodes, edges, annotate=True, title="Visualization"):
    plt.figure(dpi=128)

    plt.scatter([x for x, y in points], [y for x, y in points])

    for source, target in edges:
        node_src = nodes[source]
        node_tgt = nodes[target]
        plt.plot([node_src[0], node_tgt[0]], [node_src[1], node_tgt[1]], color='r')

    plt.scatter(nodes[:,0], nodes[:,1])
    if annotate:
        for i in range(nodes.shape[0]):
            plt.annotate(i, (nodes[i][0], nodes[i][1]))

    plt.title(title)
    plt.show()


def main(argv):
    with open("./instances/"+argv[1], 'r') as f:
        data = json.load(f)

        points = [(d['x'], d['y']) for d in data["points"]]
        nodes = {d['id']: Node(d['id'], d['x'], d['y']) for d in data["nodes"]}
        edges = [Edge(d['source'], d['target']) for d in data["edges"]]

    nodes_np = np.zeros((len(nodes),2))
    for i in range(len(nodes)):
        node=nodes[i]
        n_id = int(node.id)
        nodes_np[n_id][0]=node.x
        nodes_np[n_id][1]=node.y


    edges_np = np.zeros((len(edges),3))
    for i in range(len(edges)):
        edges_np[i][0] = edges[i].src
        edges_np[i][1] = edges[i].tgt 

    points_np = np.zeros((len(points),2))
    for i in range(len(points)):
        points_np[i] = points[i]

    edges_np = count_crossings(nodes_np, edges_np)
    init_crossings_num = edges_np.sum(axis=0)[2]/2
    print("Initial Crossing Numbers:", init_crossings_num) 


    # Run the force-directed layout algorithm
    #positions = fruchterman_reingold(nodes_np, edges_np,iterations=1000, width=points_np[:,0].max(), height=points_np[:,1].max())

    # Run the string layout function in networkx
    
    G = nx.Graph()
    G.add_edges_from(edges_np[:,:2])
    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    #nx.draw_networkx_labels(G, pos, labels={node: node for node in G.nodes()}, font_size=12)
    plt.show()

    positions = np.zeros((len(pos),2))
    
    for key, value in pos.items():
        positions[int(key)]=value

    
    width=points_np[:,0].max()
    height=points_np[:,1].max()
    
    positions[:,0]=(positions[:,0]-positions[:,0].min())/(positions[:,0].max()-(positions[:,0].min()))*width
    positions[:,1]=(positions[:,1]-positions[:,1].min())/(positions[:,1].max()-(positions[:,1].min()))*height
    
    nodes_np = nearest_neighbor(points_np, positions)

    edges_np = count_crossings(nodes_np, edges_np) # TODO
    crossings_num = edges_np.sum(axis=0)[2]/2
    print("crossings_num=",crossings_num)
    visualize(points, nodes_np, edges, title="Output Data")  
    store_output(points, nodes_np, edges_np[:,:2], "./output_6/"+argv[1])



if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)

    main(sys.argv)