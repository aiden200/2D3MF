import init
import setuptools
from setuptools import setup, find_packages


TD3MF_packages = [f"TD3MF.{pkg}" for pkg in find_packages(where='TD3MF')]
src_packages = [f"src.{pkg}" for pkg in find_packages(where='src')]
# packages=setuptools.find_packages(where="src", include=["marlin_pytorch", "marlin_pytorch.*", "fairseq"]),
all_packages = TD3MF_packages + src_packages


with open("README.md", "r", encoding="UTF-8") as file:
    long_description = file.read()

requirements = []
with open("requirements.txt", "r", encoding="UTF-8") as file:
    for line in file:
        requirements.append(line.strip())


version = init.read_version()
init.write_version(version)

setup(
    name="2D3MF",
    version=version,
    author="aiden200",
    author_email="aidenchang@gmail.com",
    description="Official pytorch implementation for 2D3MF",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aiden200/2D3MF",
    project_urls={
        "Bug Tracker": "https://github.com/aiden200/2D3MF/issues",
        "Source Code": "https://github.com/aiden200/2D3MF",
    },
    keywords=["Audio", "pytorch", "AI", "machine-learning", "video", "deep-learning", "multi-modal", "deepfake-detection"],
    packages=all_packages,
    package_data={
        "marlin_pytorch": [
            "version.txt"
        ]
    },
    python_requires=">=3.6",
    install_requires=requirements,
    license="CC BY-NC 4.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
)
