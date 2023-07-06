import hashlib
import os


def hash_url_quick(url):
    import requests
    from sdkit.utils import log

    log.debug(f"hashing url: {url}")

    def get_size():
        res = requests.get(url, stream=True)
        size = int(res.headers["content-length"])  # fail loudly if the url doesn't return a content-length header
        log.debug(f"total size: {size}")
        return size

    def read_bytes(offset: int, count: int):
        res = requests.get(url, headers={"Range": f"bytes={offset}-{offset+count-1}"})
        log.debug(f"read byte range. offset: {offset}, count: {count}, actual count: {len(res.content)}")
        return res.content

    return compute_quick_hash(
        total_size_fn=get_size,
        read_bytes_fn=read_bytes,
    )


def hash_file_quick(file_path):
    from sdkit.utils import log

    log.debug(f"hashing file: {file_path}")

    def get_size():
        size = os.path.getsize(file_path)
        log.debug(f"total size: {size}")
        return size

    def read_bytes(offset: int, count: int):
        with open(file_path, "rb") as f:
            f.seek(offset)
            bytes = f.read(count)
            log.debug(f"read byte range. offset: {offset}, count: {count}, actual count: {len(bytes)}")
            return bytes

    return compute_quick_hash(
        total_size_fn=get_size,
        read_bytes_fn=read_bytes,
    )


def compute_quick_hash(total_size_fn, read_bytes_fn):
    """
    quick-hash logic:
    - read 64k chunks from the start, middle and end, and hash them
    - start offset: 1 MB
    - middle offset: 0
    - end offset: -1 MB

    Do not use if the file size is less than 3 MB
    """
    total_size = total_size_fn()

    if total_size < 0x300000:
        all_bytes = read_bytes_fn(offset=0, count=total_size)
        return hash_bytes(all_bytes)
    else:
        start_bytes = read_bytes_fn(offset=0x100000, count=0x10000)
        middle_bytes = read_bytes_fn(offset=int(total_size / 2), count=0x10000)
        end_bytes = read_bytes_fn(offset=total_size - 0x100000, count=0x10000)

        return hash_bytes(start_bytes + middle_bytes + end_bytes)


def hash_bytes(bytes):
    return hashlib.sha256(bytes).hexdigest()
