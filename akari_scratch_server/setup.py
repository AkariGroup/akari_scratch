from setuptools import find_packages, setup

setup(
    name="akari-scratch-server",
    version="0.1.0",
    packages=find_packages(),
    author="akari",
    author_email="akari.tmc@gmail.com",
    package_data={
        "akari_scratch_server": ["py.typed"],
        "": ["data/*.blob"],
    },
    install_requires=[
        "akari-proto>=0.3.0",
        "depthai>=2.18.0",
        "fastapi==0.86.0",
        "grpcio==1.44.0",
        "numpy==1.23.4",
        "opencv-python-headless==4.7.0.72",
        "protobuf==3.19.3",
        "uvicorn==0.19.0",
    ],
    zip_safe=False,
)
