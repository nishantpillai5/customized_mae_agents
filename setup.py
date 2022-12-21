from setuptools import find_packages, setup

setup(
    name="src",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "Click",
    ],
    entry_points={
        "console_scripts": [
            "data = src.data.data:data",
            "features = src.features.features:features",
            "models = src.models.models:models",
            "visualization = src.visualization.visualization:visualization",
            "pipeline = src.pipeline:pipeline",
        ],
    },
)
