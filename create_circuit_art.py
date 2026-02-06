#!/usr/bin/env python3
import numpy as np
from PIL import Image, ImageDraw
import sys
import os
import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import read_cap
import argparse

def read_pr_output(pr_filename, verbose=False):
    """Read routing output file"""
    res = dict()
    in1 = open(pr_filename, 'r')
    total = 0
    lines = in1.readlines()
    in1.close()
    
    if verbose:
        print('Reading pr output file in memory finished... Processing...')
    
    progressbar = tqdm.tqdm(total=len(lines))
    while 1:
        if total >= len(lines):
            break
        name = lines[total].strip()
        total += 1
        progressbar.update(1)
        if name == '':
            break
        points = []
        while 1:
            line = lines[total].strip()
            total += 1
            progressbar.update(1)
            if line == '(':
                continue
            if line == ')':
                break
            r = line.strip().split(' ')
            r = [int(f) for f in r]
            points.append(r)
        if len(points) == 0:
            continue  # Skip empty nets
        res[name] = tuple(points)
    
    progressbar.close()
    return res

def create_circuit_art(cap_file, routing_file, output_file):
    """Create a beautiful circuit art image"""
    
    print("Reading capacity file...")
    data_cap = read_cap(cap_file, verbose=True)
    
    print("Reading routing output...")
    data_out = read_pr_output(routing_file, verbose=True)
    
    # Get dimensions
    height, width = data_cap['ySize'], data_cap['xSize']
    n_layers = data_cap['nLayers']
    
    print(f"Creating {width}x{height} image with {n_layers} layers")
    
    # Create high-res image (4x scale for crisp details)
    scale = 4
    img_width, img_height = width * scale, height * scale
    img = Image.new('RGB', (img_width, img_height), (0, 0, 0))  # Black background
    draw = ImageDraw.Draw(img)
    
    # Beautiful color palette for each layer
    layer_colors = [
        (255, 80, 80),    # Bright Red
        (80, 255, 80),    # Bright Green
        (80, 80, 255),    # Bright Blue
        (255, 255, 80),   # Bright Yellow
        (255, 80, 255),   # Bright Magenta
        (80, 255, 255),   # Bright Cyan
        (255, 160, 80),   # Orange
        (160, 80, 255),   # Purple
        (80, 255, 160),   # Mint
        (255, 80, 160),   # Pink
    ]
    
    # Process each net
    total_nets = len(data_out)
    for i, net in enumerate(data_out):
        if i % 5000 == 0:
            print(f"Processing net {i}/{total_nets} ({100*i/total_nets:.1f}%)")
        
        for segment in data_out[net]:
            x1, y1, z1, x2, y2, z2 = segment
            
            # Get color for this layer
            color = layer_colors[min(z1, len(layer_colors)-1)]
            
            # Scale coordinates
            sx1, sy1 = x1 * scale, y1 * scale
            sx2, sy2 = x2 * scale, y2 * scale
            
            # Draw the routing segment
            if x1 == x2 and y1 == y2:
                # Via (vertical connection) - draw as bright dot
                draw.ellipse([sx1-1, sy1-1, sx1+1, sy1+1], fill=color)
            else:
                # Wire - draw as line with some thickness
                draw.line([sx1, sy1, sx2, sy2], fill=color, width=2)
    
    print(f"Saving high-resolution image to {output_file}")
    img.save(output_file, 'PNG', quality=95)
    print(f"Circuit art saved! Image size: {img_width}x{img_height}")

if __name__ == "__main__":
    create_circuit_art(
        "test_data/ariane133_51.cap",
        "test_data/ariane_output.txt", 
        "test_data/circuit_art.png"
    )