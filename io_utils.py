import base64
import json
import logging
import pickle
import zlib
from botocore.exceptions import ClientError
from typing import Any
import numpy as np
import cv2

def encode_base64(x):
    x = pickle.dumps(x)
    x = zlib.compress(x)
    x = base64.b64encode(x).decode()
    return x


def decode_base64(s):
    s = base64.b64decode(s)
    s = zlib.decompress(s)
    s = pickle.loads(s)
    return s


def compress(x):
    x = pickle.dumps(x)
    x = zlib.compress(x)
    return x


def uncompress(s):
    s = zlib.decompress(s)
    s = pickle.loads(s)
    return s


def load_s3_object(s3_client: Any, bucket: str, key: str, encoding: str):
    logging.info(f"Loading object from s3://{bucket}/{key}")
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as e:
        logging.error(e)
        raise Exception(e)
    if encoding == "base64":
        return decode_base64(response)
    if encoding == "gz":
        return uncompress(response)
    elif encoding == "json":
        return json.loads(response)
    else:
        raise Exception(f"Unknown encoding: {encoding}")


def load_clip(s3_client, bucket, key, indices=None):
    bgr_clip = load_s3_object(s3_client, bucket, key, encoding="gz")  # TxHxWxC
    if indices is None:
        indices = np.arange(bgr_clip.shape[0])
    frames = [cv2.cvtColor(bgr_clip[i], cv2.COLOR_BGR2RGB) for i in indices]  # Length T of [HxWxC]
    return frames


def load_individual_annotations_from_s3_path(s3_client, s3_path):
    bucket, annots_filename = s3_path.replace("s3://","").split("/")
    return load_s3_object(s3_client=s3_client, bucket=bucket, key=annots_filename, encoding="json")


def save_image_to_s3(s3_client, bucket, key, image, color_format='RGB'):
    if color_format == 'BGR':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Save numpy image to s3 as jpg with cv2
    _, image = cv2.imencode(".jpg", image)
    image = image.tobytes()
    s3_client.put_object(Bucket=bucket, Key=key, Body=image)


def load_combined_annotations(s3_client, bucket, key):
    combined_annots = load_s3_object(s3_client=s3_client, 
                                              bucket=bucket, 
                                              key=f"{key}.json", 
                                              encoding="json")
    
    user_annots = load_individual_annotations_from_s3_path(s3_client, combined_annots["user_annots_s3_path"])
    tgt_annots = load_individual_annotations_from_s3_path(s3_client, combined_annots["tgt_annots_s3_path"])
    # This assert may be dropped in future.
    assert len(tgt_annots["images"]) == len(tgt_annots["annotations"]) == len(combined_annots["user2tgt_frame_indices"])
    return combined_annots, user_annots, tgt_annots