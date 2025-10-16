import os

# Hardcoded directory
directory = "/Users/tarasmusakovskyi/Desktop/fishes/video from cameras/outputFrames-Main"

# Get list of files with creation time
files = [
    (f, os.path.getctime(os.path.join(directory, f)))
    for f in os.listdir(directory)
    if os.path.isfile(os.path.join(directory, f))
]

# Sort by creation time (oldest → newest)
files.sort(key=lambda x: x[1])

# Keep only every 4th file → uniform 25%
to_keep = {files[i][0] for i in range(0, len(files), 4)}

# Delete the rest
for f, _ in files:
    if f not in to_keep:
        os.remove(os.path.join(directory, f))
        print(f"Deleted: {f}")

print(f"Kept {len(to_keep)} files out of {len(files)}")

