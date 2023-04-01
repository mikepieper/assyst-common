"""Plotting utilities for Jupyter Notebook development."""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_image(ax, im, detections, cutoff=0.3):
    # assert len(detections) == 1
    # det = detections[0]
    det = detections
    
    if 'bbox' in det:
        bbox = np.array(det['bbox'], dtype='int')
        minx, miny, maxx, maxy = bbox[:4]
        im = cv2.rectangle(im, (minx, miny), (maxx, maxy), color=(255,255,0), thickness=15)
    
    if 'keypoints' in det:
        kpts = np.array(det['keypoints'], dtype='int')
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
    
    for idx in range(len(indices)):
        i = indices[idx]
        show_image(axs[idx//4,idx%4], rgb_frames[i].copy(), detections[i], cutoff=cutoff)
