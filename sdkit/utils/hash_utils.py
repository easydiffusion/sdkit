import hashlib
import requests

def hash_url_quick(model_url):
    res = requests.get(model_url, headers={"Range": "bytes=0-2000000"}) # just the first ~2Mb
    return hash_bytes_quick(res.content)

# based on automatic1111's approach of hashing only a few bytes at the start of the model
def hash_file_quick(model_path):
    with open(model_path, 'rb') as f:
        bytes = f.read(0x110000)
        return hash_bytes_quick(bytes)

def hash_bytes_quick(bytes):
    offset = 0x100000
    num_to_read = 0x10000
    m = hashlib.sha256(bytes[offset : offset+num_to_read])
    return m.hexdigest()
