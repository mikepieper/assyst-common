import base64
import json
import logging
import pickle
import zlib
from botocore.exceptions import ClientError


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


def load_s3_object(s3_client, bucket, key, encoding: str = None):
    logging.info(f"Loading object from s3://{bucket}/{key}")
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as e:
        logging.error(e)
        return None
    if encoding == "base64":
        return decode_base64(response)
    if encoding == "gz":
        return uncompress(response)
    elif encoding == "json":
        return json.loads(response)
    else:
        raise Exception(f"Unknown encoding: {encoding}")