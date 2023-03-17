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

def draw_iter(x,y):
    plt.plot(x, y)
    plt.xlabel('Iterations')
    plt.ylabel('Number of edge crossings')
    plt.title('Evolutionary process')

    plt.show()


def on_segment(segment_start,point,segment_end):
    #determine whether point is between the segment but not the endpoints of the segment

    collinear = ((point[1] - segment_start[1]) * (segment_end[0] - segment_start[0]) ==
                 (segment_end[1] - segment_start[1]) * (point[0] - segment_start[0]))

    if not collinear:
        return False

    # Check if the point lies between the segment's endpoints
    min_x, max_x = sorted([segment_start[0], segment_end[0]])
    min_y, max_y = sorted([segment_start[1], segment_end[1]])

    # Check for strict inequality for both x and y coordinates
    return (min_x < point[0] < max_x or min_y < point[1] < max_y)

def is_collinear(p1, p2, p3):
    cross_product = np.cross(p2 - p1, p3 - p2)
    return np.allclose(cross_product, 0)

def overlap_vectorized(A1, A2, B1, B2):
    overlaps = np.zeros(len(A1), dtype=bool)

    for i, (a1, a2, b1, b2) in enumerate(zip(A1, A2, B1, B2)):
        if (on_segment(a1, b1, a2) or on_segment(a1, b2, a2) or
            on_segment(b1, a1, b2) or on_segment(b1, a2, b2)):
            overlaps[i] = True
            collinear = is_collinear(a1, a2, b1) and is_collinear(a1, a2, b2)
            if collinear:
                overlaps[i] = len(A1)
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
    
    
    # if 2 edges are overlapped, we add large weight to corresponding edges 
    # such that we will move this edge with very high probability in the next few iterations
    edges[:, 2] = crossed_counts[:n_edges] + overlaps_counts[:n_edges]

    return edges

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




def main(argv):
    iteration = int(argv[2])
    with open("./instances/"+argv[1], 'r') as f:
        data = json.load(f)
        points = [(d['x'], d['y']) for d in data["points"]]
        nodes = {d['id']: Node(d['id'], d['x'], d['y']) for d in data["nodes"]}
        edges = [Edge(d['source'], d['target']) for d in data["edges"]]

    print("number of points = ",len(points))
    print("number of nodes = ",len(nodes))
    print("number of edges = ",len(edges))


    nodes_np = np.zeros((len(nodes),2))
    newnodes = np.zeros((len(nodes),2))
    for i in range(len(nodes)):
        node=nodes[i]
        n_id = int(node.id)
        nodes_np[n_id][0]=node.x
        nodes_np[n_id][1]=node.y


    edges_np = np.zeros((len(edges),3))
    for i in range(len(edges)):
        edges_np[i][0] = edges[i].src
        edges_np[i][1] = edges[i].tgt 


    start_time = time.time()
    edges_np = count_crossings(nodes_np, edges_np)
    init_crossing_num = edges_np.sum(axis=0)[2]/2
    end_time = time.time()
    print("counting time is ", end_time - start_time)
    print("Initial Crossing Numbers:", init_crossing_num)
    #visualize(points, nodes_np, edges, title="Input Data for "+argv[1])
    crossing_num = init_crossing_num

    
    start_time = time.time()
    iter_list = []
    crossing_num_list = []

    iter_list.append(i)
    crossing_num_list.append(crossing_num)

    i=0
    kids=5
    while crossing_num>0 and i<iteration:
        stime= time.time()
        edge = random.choices(edges_np[:,:2], cum_weights=(edges_np[:,2]+1), k=1)
        src_id = int(edge[0][0])
        tgt_id = int(edge[0][1])

        edges_np = count_crossings(nodes_np, edges_np)
        crossing_num = edges_np.sum(axis=0)[2]/2
        new_crossing_num = crossing_num
        newnodes = nodes_np.copy()
        newnewnodes = nodes_np.copy()
        j=0

        #select the best individual among k
        while j<kids:
            point1,point2 = random.sample(points,2)
            while point1==point2 or (point1==nodes_np).sum(axis=1).max()==2 or (point2==nodes_np).sum(axis=1).max()==2:
                point1,point2 = random.sample(points,2)


            newnewnodes[src_id] = point1
            newnewnodes[tgt_id] = point2         

            edges_np = count_crossings(newnewnodes, edges_np)
            newnew_crossing_num = edges_np.sum(axis=0)[2]/2
            #print("newnew_crossing_num=",newnew_crossing_num)
        
            if newnew_crossing_num<new_crossing_num:
                newnodes = newnewnodes.copy()
                new_crossing_num = newnew_crossing_num
            j=j+1

        # if the selected best indivisual is better, update
        if(new_crossing_num<crossing_num):
            iter_list.append(i)
            crossing_num_list.append(crossing_num)
            nodes_np=newnodes.copy()
            crossing_num = new_crossing_num
            print("\n update iter=",i)
            print("new_crossing_num=",new_crossing_num)
            store_output(points, nodes_np, edges_np[:,:2], "./output/"+argv[1])
            visualize(points, newnodes, edges, title="Output Data")  
        i=i+1
        etime = time.time()



    end_time = time.time()
    print("initial crossing_num:", init_crossing_num)
    print("final crossing_num=",crossing_num)
    print("iterations=",i)
    print("running time is %s s" %(end_time-start_time))
    visualize(points, nodes_np, edges, title="Output Data for "+argv[1])  

    store_output(points, nodes_np, edges_np[:,:2], "./output/"+argv[1])
    draw_iter(iter_list,crossing_num_list)




if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(1)

    main(sys.argv)
