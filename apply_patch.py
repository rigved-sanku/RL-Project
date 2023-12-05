import importlib
import inspect
import os
import sys
import subprocess

def find_module_location(module_name):
    try:
        module = importlib.import_module(module_name)
        module_location = inspect.getfile(module)
        return os.path.dirname(module_location)
    except ImportError as e:
        print(f"Error finding module location: {e}")
        return None

def apply_patch(module_name, patch_file):
    module_location = find_module_location(module_name)

    if not module_location:
        print(f"Error: Could not find the location of module {module_name}")
        return

    try:
        patch_path = os.path.abspath(patch_file)
        subprocess.run(['patch', '-p1', '-d', module_location, '--input', patch_path])
        print(f"Patch applied successfully to {module_name} at {module_location}")
    except subprocess.CalledProcessError as e:
        print(f"Error applying patch: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    module_name = "inspect"
    patch_file = "patch.txt"
    apply_patch(module_name, patch_file)
