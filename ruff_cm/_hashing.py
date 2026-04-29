import hashlib


def hash_string(input_string, uid: str, algorithm='md5', truncate: int = 12):
    """Hash a string plus caller-provided uid into a short stable identifier."""
    unique_string = input_string + uid
    hash_object = hashlib.new(algorithm)
    hash_object.update(unique_string.encode('utf-8'))
    return hash_object.hexdigest()[:truncate]
