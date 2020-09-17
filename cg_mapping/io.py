from os import path

def default_mapping_dir(ff="msibi"):
    return path.join(path.dirname(__file__), f"mappings/{ff}")
