# setup.py

from setuptools import setup, find_packages

setup(
    name='mutualinfo',
    version='0.1.0',
    description='Librería para estimar la información mutua y cuantificar su incertidumbre',
    author='Jorge Santiago Rodríguez Peñalosa',
    author_email='jspenalosa@icloud.com',
    url='https://github.com/JSRP00/mutualinfo', 
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'mapie',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    python_requires='>=3.7',
)
