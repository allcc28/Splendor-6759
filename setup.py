from setuptools import setup, find_packages

setup(
    name='gym_splendor',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'gym>=0.21.0',
        'numpy',
        'pandas',
        'tqdm'
    ]
)