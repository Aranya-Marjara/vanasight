"""
Script to download required model files for VanaSight.
"""

import urllib.request
import os

def download_imagenet_labels():
    """Download ImageNet class labels."""
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    local_path = "imagenet_classes.txt"
    
    if not os.path.exists(local_path):
        print("Downloading ImageNet labels...")
        urllib.request.urlretrieve(url, local_path)
        print("Download complete!")
    else:
        print("ImageNet labels already exist.")

if __name__ == "__main__":
    download_imagenet_labels()
