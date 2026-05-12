from setuptools import setup, find_packages

setup(
    name="soul-flow",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "mediapipe",
        "numpy",
        "scipy",
        "onnxruntime",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "vision-pro=main:main",
        ],
    },
    author="Your Name",
    description="A high-fidelity spatial computing and gesture-based drawing system.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/SoulFlow",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
