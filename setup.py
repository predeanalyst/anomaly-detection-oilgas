"""
Setup script for Anomaly Detection System package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='anomaly-detection-system',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='LSTM-based anomaly detection system for offshore production assets',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/anomaly-detection-system',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'flake8>=6.0.0',
            'mypy>=1.4.0',
        ],
        'sap': [
            'pyrfc>=3.0.0',
        ],
        'cloud': [
            'boto3>=1.28.0',
            'google-cloud-storage>=2.10.0',
            'azure-storage-blob>=12.17.0',
        ],
        'monitoring': [
            'tensorboard>=2.13.0',
            'wandb>=0.15.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'anomaly-train=train:main',
            'anomaly-detect=detect:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['configs/*.yaml'],
    },
)
