from setuptools import setup, find_packages

setup(
    name="interconnector",
    version="0.1.0",
    author="Andrei Militaru",
    author_email="andrei.militaru@ist.ac.at",
    description="Laboratory automation and control toolkit for transduction experiments at ISTA",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AndreiMilitaru/interconnector",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "PyQt5",
        "peakutils",
        "matplotlib",
        "zhinst",
        "pyvisa",
        "pyyaml",
        # Add other dependencies
    ],
    license="GPLv3",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "cavity-control=interconnector.gui.cavity_control:main_entry",
        ],
    },
)
