import setuptools

setuptools.setup(
    name="ulm3d",
    version="1.2",
    description="3D ULM",
    long_description="Volumetric Ultrasound Localization Microscopy",
    long_description_content_type="text/markdown",
    author="LIB",
    license="CC BY-NC-SA 4.0",
    url="https://github.com/Lab-Imag-Bio",
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "scripts"],
    ),
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Embedded Systems",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    install_requires=[
        "numpy>=1.19.3",
        "scipy>=1.14",
        "hdf5storage>=0.1.19",
        "loguru>=0.7.2",
        "mat73>=0.63",
        "matplotlib>=3.9.0",
        "peasyTracker>=0.0.1",
        "PyYAML>=6.0.1",
        "tqdm>=4.66.4",
    ],
    python_requires=">=3.10",
)
