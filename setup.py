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
            "world = src.world.world:world",
            "adversary = src.agent.adversary:adversary",
            "player = src.agent.player:player",
            "pipeline = src.pipeline:pipeline",
        ],
    },
)
