import cv2
import numpy as np
from numpy.typing import NDArray
import os


def make_combined_clip(frames) -> NDArray:
    """Make a combined video from the source and target list of bgr frames."""
    src_image = frames[0][0]
    tgt_image = frames[1][0]
    tgt_new_height = src_image.shape[0]
    tgt_aspect_ratio = tgt_image.shape[1] / tgt_image.shape[0]  # width / height
    tgt_new_width = int(tgt_new_height * tgt_aspect_ratio)
    combined_clip = []
    for src, tgt in frames:
        combined_frame = np.concatenate([src, cv2.resize(tgt, (tgt_new_width, tgt_new_height))], axis=1)
        combined_clip.append(combined_frame)
    combined_clip = np.stack(combined_clip)

    # Cannot have odd shape in height or width.
    if combined_clip.shape[1] % 2 == 1:
        combined_clip = combined_clip[:, 1:, :, :]
    if combined_clip.shape[2] % 2 == 1:
        combined_clip = combined_clip[:, :, 1:, :]

    combined_clip = np.ascontiguousarray(np.stack(combined_clip, axis=0))  # TxHxWxC
    return combined_clip


# This is currently NOT parallelized.
def save_images_to_video(images: np.ndarray, outfile: str, fps=6):
    if os.path.exists(outfile):
        os.remove(outfile)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    images = [images[i] for i in range(images.shape[0])]
    height, width = images[0].shape[:2]
    writer = cv2.VideoWriter(outfile, fourcc, fps, (width,height), True)
    for frame in images:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

