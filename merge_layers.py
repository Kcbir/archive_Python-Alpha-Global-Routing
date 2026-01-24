#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import os

def merge_layer_videos(input_pattern, output_file):
    """Merge all layer videos into one composite video"""
    
    # Find all layer videos
    video_files = sorted(glob.glob(input_pattern))
    print(f"Found {len(video_files)} layer videos")
    
    if not video_files:
        print("No video files found!")
        return
    
    # Open all video captures
    caps = []
    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        if cap.isOpened():
            caps.append(cap)
        else:
            print(f"Failed to open {video_file}")
    
    if not caps:
        print("No valid video files!")
        return
    
    # Get video properties from first video
    width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    print(f"Creating merged video: {width}x{height} @ {fps}fps")
    
    frame_count = 0
    while True:
        frames = []
        all_valid = True
        
        # Read frame from each layer
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                all_valid = False
                break
        
        if not all_valid:
            break
        
        # Create composite frame - overlay all layers with different colors
        composite = np.zeros((height, width, 3), dtype=np.uint8)
        
        colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green  
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Cyan
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Yellow
            [128, 128, 128], # Gray
            [255, 128, 0],  # Orange
            [128, 0, 255],  # Purple
            [0, 128, 255],  # Light Blue
        ]
        
        for i, frame in enumerate(frames):
            # Convert to grayscale and create mask
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = gray < 200  # Areas with routing (darker areas)
            
            # Apply color to this layer
            if i < len(colors):
                color = colors[i]
                for c in range(3):
                    composite[:,:,c] = np.where(mask, 
                                               np.minimum(255, composite[:,:,c] + color[c] * 0.3),
                                               composite[:,:,c])
        
        out.write(composite)
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")
    
    # Cleanup
    for cap in caps:
        cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Created merged video: {output_file} ({frame_count} frames)")

if __name__ == "__main__":
    merge_layer_videos("test_data/routing_heatmap_*.mp4", "test_data/routing_3d_composite.mp4")