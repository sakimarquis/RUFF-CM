from setuptools import setup, find_packages


setup(
    name='ruff_cm',
    version='0.2',
    author='Hua-Dong Xiong',
    author_email='hdx@arizona.edu',
    description='Reusable Utility Function For Computational Modeling',
    long_description=open('README.md').read(),
    packages=find_packages(),
    license='LICENSE.txt',
    install_requires=[
        "numpy",
        "pyyaml",
        "ruamel.yaml",
        "submitit",
    ],
)
