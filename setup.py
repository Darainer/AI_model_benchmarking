from setuptools import setup, find_packages

setup(
    name="ai_model_benchmarking",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "requests>=2.26.0",
        "huggingface_hub>=0.12.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "onnx-gpu": ["onnxruntime-gpu>=1.12.0"],
        "onnx-cpu": ["onnxruntime>=1.12.0"],
        "torch": ["torch>=1.11.0", "torchvision>=0.12.0"],
        "yolo": ["ultralytics>=8.0.0"],
    },
)
