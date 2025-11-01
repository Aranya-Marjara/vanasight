# VanaSight: Forest Vision 🌳👁️

*From Pixels to Perception - A complete computer vision pipeline*

## 🚀 Quick Start

```bash
# Install from source
git clone https://github.com/Aranya-Marjara/vanasight.git
cd vanasight
pip install -e .

# Use CLI
vanasight --input your_image.jpg --output results.jpg

# Use Python API
from vanasight import VanaSight
pipeline = VanaSight()
results = pipeline.run_pipeline("image.jpg", "output.jpg")

```bash
# Update utils.py
cat > vanasight/utils.py << 'EOF'
"""
Utility functions for VanaSight package.
"""

import os
import requests
from typing import Optional

def download_file(url: str, local_path: str) -> bool:
    """Download file from URL to local path."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def ensure_directory(path: str):
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)

def get_file_extension(url: str) -> str:
    """Extract file extension from URL."""
    return os.path.splitext(url.split('/')[-1])[1]
