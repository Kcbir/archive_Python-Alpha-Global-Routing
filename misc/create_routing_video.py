# coding: utf-8
__author__ = 'Roman Solovyev: https://github.com/ZFTurbo'

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import cv2
import argparse
import numpy as np
import time


CACHE_PATH = CURRENT_PATH + 'cache/'
if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)


def create_video(image_list, out_file, fps, codec):
    height, width = image_list[0].shape[:2]
    # fourcc = cv2.VideoWriter_fourcc(*'DIB ')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    # fourcc = -1
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(out_file, fourcc, fps, (width, height), True)

    for i in range(image_list.shape[0]):
        im = image_list[i]
        if len(im.shape) == 2:
            im = np.stack((im, im, im), axis=2)
        video.write(im.astype(np.uint8))
    cv2.destroyAllWindows()
    video.release()


def read_pr_output(pr_filename, verbose=False):
    start_time = time.time()
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
            print('Zero points for {}...'.format(name))
            exit()

        res[name] = tuple(points)

    progressbar.close()
    if verbose:
        print('Reading pr output time: {:.2f} sec'.format(time.time() - start_time))
    return res


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-cap", required=True, type=str, help="Location of cap file")
    parser.add_argument("-net", required=True, type=str, help="Location of net file")
    parser.add_argument("-pr", required=True, type=str, help="Location of PR output file")
    parser.add_argument("-mp4", required=True, type=str, help="Name of MP4 to store results")
    args = parser.parse_args()

    print('Read output: {}'.format(args.pr))
    data_out = read_pr_output(args.pr, verbose=True)
    print('Read cap: {}'.format(args.cap))
    data_cap = read_cap(args.cap, verbose=True)
    print('Read net: {}'.format(args.net))
    data_net = read_net(args.net, verbose=True)

    matrix = np.round(data_cap['cap']).astype(np.int32)
    via_matrix = np.zeros(data_cap['cap'].shape, dtype=np.int16)

    max_vals = []
    for i in range(matrix.shape[0]):
        max_vals.append(matrix[i].max())

    print(matrix.shape, matrix.min(), matrix.max(), matrix.mean())
    
    # Initialize VideoWriters for all layers
    height, width = matrix.shape[1], matrix.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video_writers = []
    for layer in range(matrix.shape[0]):
        video_path = args.mp4[:-4] + '_{}.mp4'.format(layer)
        writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height), True)
        video_writers.append(writer)
    
    print(f"Created {len(video_writers)} video writers")
    
    total = 0
    max_vals_array = np.array(max_vals).astype(np.float32)
    
    for net in tqdm.tqdm(data_out):
        for r in data_out[net]:
            x1, y1, z1, x2, y2, z2 = r
            if z1 == z2:
                if y1 != y2:
                    matrix[z1:z1 + 1, y1:y2, x1:x1 + 1] -= 1
                else:
                    matrix[z1:z1 + 1, y1:y1 + 1, x1:x2] -= 1
        
        total += 1
        if total % 400 == 0:
            # Process each layer immediately and write to video
            for layer in range(matrix.shape[0]):
                # Create frame for this layer only
                layer_data = matrix[layer]
                normalized = 255.0 * layer_data / max_vals_array[layer]
                
                # Create RGB frame
                frame = np.stack([normalized, normalized, normalized], axis=-1)
                frame[:, :, 0][frame[:, :, 0] < 0] = 0
                frame[:, :, 1][frame[:, :, 1] < 0] = 0  
                frame[:, :, 2][frame[:, :, 2] < 0] = 255
                
                # Write frame immediately and drop reference
                video_writers[layer].write(frame.astype(np.uint8))
                del frame  # Explicit cleanup
    
    # Close all video writers
    for writer in video_writers:
        writer.release()
    cv2.destroyAllWindows()
    print('Overall time: {:.2f} sec'.format(time.time() - start_time))

