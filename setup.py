from setuptools import setup, find_packages

setup(
    name='lensit',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'sncosmo',
        'lenstronomy',
        'tqdm'
    ],
    author='Ana Sagués Carracedo',
    author_email='anita.17.eny@gmail.com',
    description='A package for analyzing gravitationally lensed supernova in wide field surveys.',
    keywords='astronomy',
)