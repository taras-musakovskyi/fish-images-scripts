import os

# Hardcoded directory
directory = "/Users/tarasmusakovskyi/Desktop/fishes/video from cameras/outputFrames-Main/2"

for filename in os.listdir(directory):
    old_path = os.path.join(directory, filename)
    if os.path.isfile(old_path):  # skip subdirectories
        new_filename = f"2-{filename}"
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)
        print(f'Renamed: {filename} -> {new_filename}')
