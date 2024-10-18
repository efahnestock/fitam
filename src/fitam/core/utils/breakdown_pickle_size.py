import pickle
import sys
from collections import namedtuple
from collections.abc import Mapping, Iterable

def human_readable_size(size, decimal_places=2):
    """Converts a size in bytes to a human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"

def get_size(obj, seen=None):
    """Recursively finds size of objects in bytes."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def is_namedtuple(obj):
    """Check if an object is an instance of a named tuple."""
    return isinstance(obj, tuple) and hasattr(obj, '_fields')

def print_size(obj, prefix='', depth=0, max_depth=None):
    """Prints the size of nested objects, including keys or fields names."""
    if max_depth is not None and depth > max_depth:
        return
    human_readable_size_ = human_readable_size(get_size(obj))
    print(f"{prefix}{type(obj).__name__}: {human_readable_size_}")
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            print_size(v, f"{prefix}{k}: ", depth + 1, max_depth)
    elif is_namedtuple(obj):
        for field in obj._fields:
            print_size(getattr(obj, field), f"{prefix}{field}: ", depth + 1, max_depth)
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, bytearray)):
        for item in obj:
            print_size(item, prefix + '  ', depth + 1, max_depth)

def main(pickle_file_path, max_depth):
    try:
        with open(pickle_file_path, 'rb') as f:
            loaded_obj = pickle.load(f)
            print("Inspecting objects in pickle file...")
            print_size(loaded_obj, max_depth=max_depth)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Inspect and report sizes of objects in a pickle file with depth control.')
    parser.add_argument('pickle_file', type=str, help='Path to the pickle file to inspect.')
    parser.add_argument('--max_depth', type=int, default=1, help='Maximum depth to traverse inside the objects. Use -1 for no limit.')
    args = parser.parse_args()
    # Convert max_depth from -1 to None if no limit is set
    max_depth = None if args.max_depth == -1 else args.max_depth
    main(args.pickle_file, max_depth)
