# coding: utf-8
__author__ = 'Roman Solovyev: https://github.com/ZFTurbo'

import tqdm
import os
import numpy as np
import time

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'

def read_cap(path, verbose=False):
    start_time = time.time()
    res = dict()
    in1 = open(path, 'r')

    line = in1.readline()
    nLayers, xSize, ySize = line.strip().split(" ")
    nLayers = int(nLayers)
    xSize = int(xSize)
    ySize = int(ySize)

    line = in1.readline()
    arr = line.strip().split(" ")
    unit_length_wire_cost = float(arr[0])
    unit_via_cost = float(arr[1])
    unit_length_short_costs = []
    for i in range(len(arr) - 2):
        unit_length_short_costs.append(float(arr[i+2]))

    line = in1.readline()
    arr = line.strip().split(" ")
    horizontal_GCell_edge_lengths = []
    for i in range(len(arr)):
        horizontal_GCell_edge_lengths.append(int(arr[i]))

    line = in1.readline()
    arr = line.strip().split(" ")
    vertical_GCell_edge_lengths = []
    for i in range(len(arr)):
        vertical_GCell_edge_lengths.append(int(arr[i]))

    layerNames = []
    layerDirections = []
    layerMinLengths = []
    cap = np.zeros((nLayers, ySize, xSize), dtype=np.float32)
    for i in range(nLayers):
        line = in1.readline()
        name, direction, min_length = line.strip().split(" ")
        layerNames.append(name)
        layerDirections.append(int(direction))
        layerMinLengths.append(int(min_length))
        for j in range(ySize):
            line = in1.readline()
            arr = line.strip().split(" ")
            arr = np.array(arr, dtype=np.float32)
            cap[i, j, :] = arr

    res['nLayers'] = nLayers
    res['xSize'] = xSize
    res['ySize'] = ySize
    res['unit_length_wire_cost'] = unit_length_wire_cost
    res['unit_via_cost'] = unit_via_cost
    res['unit_length_short_costs'] = unit_length_short_costs
    res['horizontal_GCell_edge_lengths'] = horizontal_GCell_edge_lengths
    res['vertical_GCell_edge_lengths'] = vertical_GCell_edge_lengths
    res['layerNames'] = layerNames
    res['layerDirections'] = layerDirections
    res['layerMinLengths'] = layerMinLengths

    # direction dict
    dd = dict()
    for i in range(len(layerDirections)):
        dd[res['layerNames'][i]] = res['layerDirections'][i]

    # metal level
    ml = dict()
    for i in range(len(layerDirections)):
        ml[res['layerNames'][i]] = i

    res['cap'] = cap
    res['dir'] = dd
    res['level'] = ml

    if verbose:
        print('Reading caps time: {:.2f} sec'.format(time.time() - start_time))
    in1.close()
    return res


def read_net(path, verbose=False):
    start_time = time.time()
    res = dict()
    in1 = open(path, 'r')

    total = 0
    lines = in1.readlines()
    in1.close()
    if verbose:
        print('Reading net file in memory finished... Processing...')

    progressbar = tqdm.tqdm(total=len(lines))
    while 1:
        if total >= len(lines):
            break
        name = lines[total].strip(); total += 1; progressbar.update(1)
        if name == '':
            break
        points = []
        while 1:
            line = lines[total].strip(); total += 1; progressbar.update(1)
            if line == '(':
                continue
            if line == ')':
                break
            # Parse line: pin_name, capacitance, [(layer, x, y), ...]
            # Split by comma, but be careful with commas inside brackets
            parts = line.split(', ', 2)  # Split into at most 3 parts
            if len(parts) == 3:
                pin_name = parts[0]
                capacitance = float(parts[1])
                coordinates = eval(parts[2])
                r = (pin_name, capacitance, coordinates)
            else:
                # Fallback to original eval for backward compatibility
                r = eval(line)
            points.append(r)
        if len(points) == 0:
            print('Zero points for {}...'.format(name))
            exit()

        res[name] = tuple(points)

    progressbar.close()
    if verbose:
        print('Reading nets time: {:.2f} sec'.format(time.time() - start_time))
    return res



