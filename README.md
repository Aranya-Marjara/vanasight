# 🌿 VanaSight: Forest Vision

*From Pixels to Perception — A Complete Computer Vision Pipeline*

> 🧠 **Developed and Tested on Gentoo Linux**
>
> This project was fully built and tested on Gentoo Linux — though it should work on most systems with Python 3.8+, the setup and dependencies are verified only on Linux environments.

---

## 🚀 Installation

### 🧪 Gentoo Linux (Developed & Tested Platform)
```bash
git clone https://github.com/Aranya-Marjara/VanaSight.git
cd VanaSight
pip install -e .
```

### 🐧 Ubuntu / Debian
```bash
sudo apt update
sudo apt install python3-pip python3-venv -y
git clone https://github.com/Aranya-Marjara/VanaSight.git
cd VanaSight
pip install -e .
```

### 🪟 Windows
```bash
# Install Python 3.8+ from python.org first
git clone https://github.com/Aranya-Marjara/VanaSight.git
cd VanaSight
pip install -e .
```

### 🍎 macOS
```bash
# Install Homebrew first
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python
git clone https://github.com/Aranya-Marjara/VanaSight.git
cd VanaSight
pip install -e .
```

---

## ⚡ Usage

```bash
# Basic usage
vanasight --input image.jpg --output results.jpg

# From URL
vanasight --input "https://picsum.photos/800/600" --output output.jpg

# Force CPU mode
vanasight --input image.jpg --device cpu
```

---

## Real-World Applications

> **Educational Focus**
> 
> VanaSight demonstrates a complete computer vision pipeline for learning purposes. It shows how images transform through different CV stages. It was never built to compete

| Stage | What It Actually Does | Technology |
|-------|----------------------|------------|
| Image Loading | Loads local files or URLs with fallbacks | OpenCV, Requests |
| Noise Removal | Reduces image grain and noise | FastNlMeansDenoising |
| Contrast Enhancement | Improves image contrast locally | CLAHE Algorithm |
| Sharpening | Enhances edges and details | Convolution Filters |
| AI Classification | Identifies image content (1000 categories) | ResNet-18 |
| Object Detection | Finds prominent shapes/edges in image | Canny + Contour analysis |
| Artistic Filters | Applies style-based image transformations | PIL Filters + Effects |

### Actual Use Cases:
- **Education**: Learn complete CV pipeline from input to output
- **Prototyping**: Test image enhancement techniques
- **Demonstrations**: Show how different CV stages work
- **Experimentation**: Modify and extend pipeline components

## 🧩 How It Works — Flowchart

```text
Input Image
    │
    ▼
[Load Image Module]
 ├── Check local file
 ├── If fails → try URL
 └── If fails → generate synthetic image
    │
    ▼
[Enhancement Pipeline]
 ├── Denoise → CLAHE → Sharpen
 ├── Boost colors → Gamma correct
    │
    ▼
[AI Classification]
 ├── Preprocess (resize + normalize)
 ├── ResNet-18 inference
 └── Return top-3 predictions
    │
    ▼
[Object Detection]
 ├── Convert grayscale → Canny
 ├── Contour filter + labeling
 └── Draw bounding boxes
    │
    ▼
[AI Art Generator]
 ├── Analyze class → Apply style:
 │     ├─ Nature → Dreamy
 │     ├─ Portraits → Warm focus
 │     └─ Others → High contrast
 └── Add vignette + texture
    │
    ▼
[Visualization]
 └── Combine all in 2×2 grid → Save result.jpg
```

---

## ⚙️ Technical Stack

| Component | Library / Framework |
| ---------- | ------------------ |
|  AI Model | PyTorch (ResNet-18) |
|  Image I/O | OpenCV, PIL |
|  CLI | argparse |
|  Packaging | setuptools |
|  Denoising | OpenCV FastNlMeans |
|  Enhancement | CLAHE, Gamma |
|  Detection | Canny + Contours |
|  Style Transfer | Custom filters |

---

## Developer Notes

- Built and tested **exclusively on Gentoo Linux**
- Debugging was done with the help of AI (You should not expect more from a self-taught Python programmer)
- GPU support optional (PyTorch auto-detects CUDA)
- Modular pipeline: every stage can run independently
- Ideal for demos, CV research prototypes, or AI art workflows
---

## 🐾 Aranya-Marjara Organization

> “Where Code Meets the Wilderness.”

A non-profit, open-source collective building ethical AI ecosystems —  
balancing the grace of the forest and the wisdom of the wild.

🌐 [https://github.com/Aranya-Marjara](https://github.com/Aranya-Marjara)

---
