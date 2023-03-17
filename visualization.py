import json
import sys
from collections import namedtuple
from itertools import permutations
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import time
import pdb
import random

Node = namedtuple('Node', ['id', 'x', 'y'])
Edge = namedtuple('Edge', ['src', 'tgt'])

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
    with open("./output_final/"+argv[1], 'r') as f:
        data = json.load(f)

        points = [(d['x'], d['y']) for d in data["points"]]
        nodes = {d['id']: Node(d['id'], d['x'], d['y']) for d in data["nodes"]}
        edges = [Edge(d['source'], d['target']) for d in data["edges"]]
    print("edges num = ",len(nodes))
    nodes_np = np.zeros((len(nodes),2))
    for i in range(len(nodes)):
        node=nodes[i]
        n_id = int(node.id)
        nodes_np[n_id][0]=node.x
        nodes_np[n_id][1]=node.y

    print("num of edges = ",len(edges))
    edges_np = np.zeros((len(edges),3))
    for i in range(len(edges)):
        edges_np[i][0] = edges[i].src
        edges_np[i][1] = edges[i].tgt 



    visualize(points, nodes_np, edges, title="Output Data for "+argv[1])  


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)

    main(sys.argv)
