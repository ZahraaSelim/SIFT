# SIFT Feature Matching and Object Detection

This Python script performs feature matching and object detection using the SIFT (Scale-Invariant Feature Transform) algorithm.

## Directory Structure

The directory structure should look like this:

```
/project-directory 
    /queries 
        lighter.jpg 
    /outputs 
    README.md
    sift.py 
    target.jpg
```

- `queries/`: Folder where the query (object) images are stored.
- `outputs/`: Folder where the result images (matched and detected images) will be saved.
- `README.md`: This readme file.
- `sift.py`: The Python script for feature matching and object detection.
- `target.jpg`: The target scene image located in the main directory.

## Usage
To use the script, you need to provide one command-line argument:

- --query: Path to the query image (the object image)

### Example Command
```
python sift.py --query lighter.jpg
```
