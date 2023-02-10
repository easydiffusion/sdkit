"""
    Hashing utilities
"""

import os
import hashlib
import requests


def hash_url_quick(url):
    """
        Compute a quick hash of a url.
    """
    from sdkit.utils import log
    log.debug('hashing url: %s', url)

    def get_size():
        res = requests.get(url, stream=True)
        # fail loudly if the url doesn't return a content-length header
        size = int(res.headers['content-length'])
        log.debug('total size: %s', size)
        return size

    def read_bytes(offset: int, count: int):
        res = requests.get(
            url,
            headers={"Range": f"bytes={offset}-{offset+count-1}"},
            timeout=1800  # Maybe this should be more reasonable?
        )
        log.debug('read byte range. offset: %s, count: %s, actual count: %s',
                  offset, count, len(res.content))
        return res.content

    return compute_quick_hash(
        total_size_fn=get_size,
        read_bytes_fn=read_bytes,
    )


def hash_file_quick(file_path):
    """
        Compute a quick hash of a file.
    """
    from sdkit.utils import log
    log.debug('hashing file: %s', file_path)

    def get_size():
        size = os.path.getsize(file_path)
        log.debug('total size: %s', size)
        return size

    def read_bytes(offset: int, count: int):
        with open(file_path, 'rb') as fp:
            fp.seek(offset)
            _bytes = fp.read(count)
            log.debug(
                'read byte range. offset: %s, count: %s, actual count: %s',
                offset, count, len(_bytes)
            )
            return _bytes

    return compute_quick_hash(
        total_size_fn=get_size,
        read_bytes_fn=read_bytes,
    )


def compute_quick_hash(total_size_fn, read_bytes_fn):
    '''
    quick-hash logic:
    - read 64k chunks from the start, middle and end, and hash them
    - start offset: 1 MB
    - middle offset: 0
    - end offset: -1 MB

    Do not use if the file size is less than 3 MB
    '''
    total_size = total_size_fn()
    one_mb = 0x100000
    start_bytes = read_bytes_fn(offset=one_mb, count=one_mb)
    middle_bytes = read_bytes_fn(offset=int(total_size/2), count=one_mb)
    end_bytes = read_bytes_fn(offset=total_size - one_mb, count=one_mb)

    return hash_bytes(start_bytes + middle_bytes + end_bytes)


def hash_bytes(bytes):
    """
        Compute a hash of a byte array.
    """
    return hashlib.sha256(bytes).hexdigest()
