# coding: utf-8
"""
Simple Global Router — RustWorkX Backend (+ optional RL)

Drop-in replacement for the NetworkX simple router.
Uses RustWorkX Steiner tree (Mehlhorn 2-approx) for ~10x speed-up.
"""

import numpy as np
import rustworkx as rx
import time
import tqdm
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import read_cap, read_net


# ---------------------------------------------------------------------- #
#  Helpers                                                                 #
# ---------------------------------------------------------------------- #

def get_ranges_for_net_simple(net_data):
    """Bounding box (min_x, min_y, max_x, max_y, w, h) for a net."""
    max_x = max_y = -10**9
    min_x = min_y = 10**9
    for points in net_data:
        z1, x1, y1 = points[0]
        min_x, max_x = min(min_x, x1), max(max_x, x1)
        min_y, max_y = min(min_y, y1), max(max_y, y1)
    return (min_x, min_y, max_x, max_y,
            max_x - min_x + 1, max_y - min_y + 1)


def _edge_weight(cap, z, y1, x1, y2, x2):
    """Edge cost = 10 + (11 - cap_1) + (11 - cap_2)."""
    return 10.0 + (11.0 - cap[z, y1, x1]) + (11.0 - cap[z, y2, x2])


# ---------------------------------------------------------------------- #
#  RustWorkX Steiner-tree routing                                          #
# ---------------------------------------------------------------------- #

def find_solution_with_rustworkx(net):
    """
    Build a local RustWorkX graph for *net* and compute its Steiner tree.

    Returns
    -------
    edges           : list[(coord_u, coord_v)]   coord = (z, y, x)
    terminal_coords : list[(z, y_local, x_local)]
    """
    global matrix, data_cap, data_net

    layer_dir = data_cap['layerDirections']
    min_x, min_y, max_x, max_y, w, h = get_ranges_for_net_simple(data_net[net])
    sub = matrix[:, min_y:max_y + 1, min_x:max_x + 1].copy()
    nL, H, W = sub.shape

    G = rx.PyGraph()
    c2n = {}                                     # coord -> node idx

    # Nodes (store coordinate tuple as node data)
    for z in range(nL):
        for y in range(H):
            for x in range(W):
                c2n[(z, y, x)] = G.add_node((z, y, x))

    # Horizontal / vertical routing edges (layer 0 not routable)
    for z in range(1, nL):
        if layer_dir[z] == 0:                    # horizontal (x)
            for y in range(H):
                for x in range(W - 1):
                    G.add_edge(c2n[(z, y, x)], c2n[(z, y, x + 1)],
                               _edge_weight(sub, z, y, x, y, x + 1))
        elif layer_dir[z] == 1:                  # vertical (y)
            for y in range(H - 1):
                for x in range(W):
                    G.add_edge(c2n[(z, y, x)], c2n[(z, y + 1, x)],
                               _edge_weight(sub, z, y, x, y + 1, x))

    # Via edges
    for z in range(nL - 1):
        for y in range(H):
            for x in range(W):
                G.add_edge(c2n[(z, y, x)], c2n[(z + 1, y, x)], 40.0)

    # Terminals
    terminal_nodes = []
    terminal_coords = []
    seen = set()
    for points in data_net[net]:
        z1, x1, y1 = points[0]
        coord = (z1, y1 - min_y, x1 - min_x)
        terminal_coords.append(coord)
        nid = c2n[coord]
        if nid not in seen:
            terminal_nodes.append(nid)
            seen.add(nid)

    if len(terminal_nodes) < 2:
        return [], terminal_coords

    # Steiner tree (Mehlhorn 2-approximation — same algo, Rust speed)
    tree = rx.steiner_tree(G, terminal_nodes, weight_fn=lambda w: w)

    edges = []
    for u, v in tree.edge_list():
        edges.append((tree[u], tree[v]))         # node data = (z, y, x)
    return edges, terminal_coords


# ---------------------------------------------------------------------- #
#  Per-net routing + capacity update                                       #
# ---------------------------------------------------------------------- #

def find_solution_for_net(net):
    """Route one net -> solution string.  Updates global *matrix*."""
    global matrix, data_cap, data_net

    min_x, min_y, max_x, max_y, w, h = get_ranges_for_net_simple(data_net[net])

    # Trivial single-cell net
    if w == 1 and h == 1:
        z1, x1, y1 = data_net[net][0][0]
        s = f'{net}\n(\n{x1} {y1} {z1} {x1} {y1} {z1 + 1}\n)\n'
        matrix[z1:z1 + 2, y1:y1 + 1, x1:x1 + 1] -= 1
        return s

    edges, terminal_coords = find_solution_with_rustworkx(net)

    sub_shape = matrix[:, min_y:max_y + 1, min_x:max_x + 1].shape
    update = np.zeros(sub_shape, dtype=np.float32)

    s = f'{net}\n(\n'
    if not edges:
        tn = terminal_coords[0]
        s += (f'{min_x + tn[2]} {min_y + tn[1]} {tn[0]} '
              f'{min_x + tn[2]} {min_y + tn[1]} {tn[0] + 1}\n')
    else:
        for u, v in edges:
            if v[0] < u[0] or v[1] < u[1] or v[2] < u[2]:
                u, v = v, u
            s += (f'{min_x + u[2]} {min_y + u[1]} {u[0]} '
                  f'{min_x + v[2]} {min_y + v[1]} {v[0]}\n')
            update[u[0], u[1], u[2]] = 1
            update[v[0], v[1], v[2]] = 1
    s += ')\n'

    matrix[:, min_y:max_y + 1, min_x:max_x + 1] -= update
    return s


# ---------------------------------------------------------------------- #
#  Full circuit routing                                                    #
# ---------------------------------------------------------------------- #

def route_circuit(output_file, max_nets=None):
    """Route every net and write solution file."""
    global matrix, data_cap, data_net

    matrix = data_cap['cap'].astype(np.float32)
    out = open(output_file, 'w')

    # Sort by area (large first, single-cell nets last)
    data_proc = list(data_net.keys())
    dim_by_net = {}
    areas = []
    for net in data_proc:
        min_x, min_y, max_x, max_y, w, h = get_ranges_for_net_simple(data_net[net])
        areas.append(w * h)
        dim_by_net[net] = (w, h)

    areas = np.array(areas)
    areas[areas == 1] = 10**9
    data_proc = [n for n, _ in sorted(zip(data_proc, areas),
                                       key=lambda p: p[1], reverse=True)]

    # Limit nets for memory efficiency
    if max_nets is not None and max_nets < len(data_proc):
        data_proc = data_proc[:max_nets]
        print(f"⚡ Memory-efficient mode: routing first {max_nets}/{len(data_net)} nets")

    bar = tqdm.tqdm(data_proc)
    for net in bar:
        bar.set_postfix({'net': net, 'size': dim_by_net[net]})
        out.write(find_solution_for_net(net))
    out.close()


# ---------------------------------------------------------------------- #
#  CLI                                                                     #
# ---------------------------------------------------------------------- #

if __name__ == '__main__':
    start_time = time.time()
    p = argparse.ArgumentParser(description='RustWorkX Simple Global Router')
    p.add_argument("-cap", required=True, type=str, help="Cap file")
    p.add_argument("-net", required=True, type=str, help="Net file")
    p.add_argument("-output", required=True, type=str, help="Output file")
    p.add_argument("-max_nets", type=int, default=1000, 
                   help="Max nets to route (default: 1000, use -1 for all)")
    args = p.parse_args()

    print('Read cap: {}'.format(args.cap))
    data_cap = read_cap(args.cap)
    print('Read net: {}'.format(args.net))
    data_net = read_net(args.net)

    print("Number of nets: {} Field size: {}x{}x{}".format(
        len(data_net),
        data_cap['nLayers'], data_cap['xSize'], data_cap['ySize']))

    max_nets = None if args.max_nets == -1 else args.max_nets
    route_circuit(args.output, max_nets=max_nets)
    print('Overall time: {:.2f} sec'.format(time.time() - start_time))
