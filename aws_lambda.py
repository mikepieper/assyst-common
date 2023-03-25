import json
import boto3
import numpy as  np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from math import ceil
from .io_utils import compress

s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda', region_name='us-east-1')


def invoke_detector(function_name, clip):
    """
        Ex. Usage:
            inputs = np.stack([img for img in inputs])
            dets = invoke_detector('person-detection', inputs)
    """
    block = 4
    T = clip.shape[0]
    inputs = [clip[block*i:block*(i+1)] for i in range(ceil(T/block))]
    fcn = partial(invoke_detector_once, function_name)
    with ThreadPoolExecutor(max_workers=8) as executor:
        preds = executor.map(fcn, inputs)
    preds = np.vstack([x for x in preds])
    return preds


def invoke_detector_once(function_name, clip):
    """
        function_names:
            pose-detection
    """
    assert clip.ndim == 4
    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType='RequestResponse',
        Payload=json.dumps({"images": compress(clip)})
    )
    result = json.loads(response['Payload'].read())
    preds = np.array(result['body']['preds'], dtype=np.float32)
    return preds
    
