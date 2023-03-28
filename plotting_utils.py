"""Plotting utilities for Jupyter Notebook development."""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_image(ax, im, detections, cutoff=0.3):
    assert len(detections) == 1
    det = detections[0]
    
    if 'bbox' in det:
        bbox = det['bbox']
        minx = int(bbox[0])
        miny = int(bbox[1])
        maxx = int(bbox[2])
        maxy = int(bbox[3])
        im = cv2.rectangle(im,
                        (minx, miny),
                        (maxx, maxy),
                        color=(255,255,0),
                        thickness=15)
    
    if 'keypoints' in det:
        kpts = det['keypoints']
        high_conf_kpts = kpts[kpts[:,2] >= cutoff]
        low_conf_kpts = kpts[kpts[:,2] < cutoff]
        ax.scatter(high_conf_kpts[:,0], high_conf_kpts[:,1], s=5, c='b')
        ax.scatter(low_conf_kpts[:,0], low_conf_kpts[:,1], s=5, c='r')
        
    ax.imshow(im)

    
def show_detections(rgb_frames, detections, cutoff=0.3):
    if len(rgb_frames) < 16:
        indices = np.arange(len(rgb_frames))
    else:
        indices = np.linspace(0, len(rgb_frames)-1, 16).astype(int)

    n_rows = len(indices)//4
    fig, axs = plt.subplots(n_rows,4, figsize=(20,5*n_rows))
    
    for i in indices:
        show_image(axs[i//4,i%4], rgb_frames[i].copy(), detections[i], cutoff=cutoff)
