from setuptools import setup, find_packages

setup(
    name="voidformer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0,<2.15.0",
        "numpy>=1.24.0,<3.0.0",
        "scipy>=1.10.0,<2.0.0",
        "pyyaml>=6.0,<7.0",
        "matplotlib>=3.7.0,<4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "qiskit>=1.0.0",
            "qiskit-aer>=0.13.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "qiskit": [
            "qiskit>=1.0.0",
            "qiskit-aer>=0.13.0",
        ],
    },
)
