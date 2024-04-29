from setuptools import setup, find_packages

setup(
    name="mydockerinstaller",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": ["mydockerinstaller = mydockerinstaller.installer:main"]
    },
)
