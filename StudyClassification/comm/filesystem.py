
import os

def solve_filename_conflict(path: str) -> str:
    if not os.path.exists(path):
        return path
    
    base, ext = os.path.splitext(path)
    counter = 1
    new_path = f"{base}_{counter}{ext}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base}_{counter}{ext}"
    return new_path

def solve_foldername_conflict(path: str) -> str:
    if not os.path.exists(path):
        return path
    
    base = path
    counter = 1
    new_path = f"{base}_{counter}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base}_{counter}"
    return new_path
