from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
requirements_path = "requirements.txt"

with open(requirements_path, "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()


setup(
    name="GPPS",  
    version="0.1.0",
    description="Scalable h-adaptive Gaussian Process Probabilistic Solver for time-independent and time-dependent systems",
    long_description_content_type="text/markdown",

    author="sk",
    url="https://github.com/savnkr/GPPS",
    license="MIT",

    packages=find_packages(),
    install_requires=requirements,

    python_requires=">=3.12",
    include_package_data=True,
)
