from setuptools import find_packages, setup


setup(
    name="ruff_cm",
    version="0.3.0",
    author="Hua-Dong Xiong",
    author_email="hdx@arizona.edu",
    description="Reusable Utility Functions for Computational Modeling",
    long_description=open("README.md", encoding="utf-8").read(),
    packages=find_packages(),
    license="LICENSE.txt",
    install_requires=[
        "numpy",
        "pandas",
        "pyyaml",
        "ruamel.yaml",
        "submitit",
        "torch",
        "tensorboard",
        "wandb",
        "matplotlib",
        "scipy",
    ],
)
