from setuptools import setup, find_packages

setup(
    name="wahkon",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.1",
        "matplotlib>=3.6",
        "tqdm>=4.64",
        "sympy>=1.11",
    ],
)
