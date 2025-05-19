import os
from setuptools import setup, find_packages

__version__ = "0.3.7"

setup(
    name="lanro_gym",
    description="Gymnasium multi-goal environments for goal-conditioned and language-conditioned deep reinforcement learning build with PyBullet",
    author="Frank Röder",
    author_email="frank.roeder@tuhh.de",
    license="MIT",
    url="https://github.com/frankroeder/lanro-gym",
    packages=[package for package in find_packages() if package.startswith("lanro_gym")],
    package_data={ "lanro_gym": ["VERSION"] },
    include_package_data=True,
    version=__version__,
    install_requires=["gymnasium==0.29.1", "pybullet", "numpy"],
    extras_require={
        "dev": ["pytest", "ruff", "ipdb", "glfw"],
        "extra": [
            "stable-baselines3 @ git+https://github.com/DLR-RM/stable-baselines3.git"
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)