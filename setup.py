from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "VanaSight: Forest Vision - From Pixels to Perception"

try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    requirements = [
        "torch>=1.9.0",
        "torchvision>=0.10.0", 
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "numpy>=1.19.0",
        "requests>=2.25.0"
    ]

setup(
    name="vanasight",
    version="1.0.0",
    author="Aranya-Marjara",
    description="VanaSight: Forest Vision - From Pixels to Perception",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Aranya-Marjara/vanasight",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "vanasight=vanasight.pipeline:main",
        ],
    },
)
