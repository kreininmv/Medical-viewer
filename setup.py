from setuptools import setup, find_packages

setup(
    name='medical',
    version='0.1.1',
    author='Kreinin Matvei',
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0,<2.0.0",
        "opencv-python-headless>=4.8",
        "matplotlib>=3.7.2",
        "IPython>=8.14.0",
        "ipywidgets>=8.1.0",
        "connected-components-3d>=3.12.0",
    ],
)