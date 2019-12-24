# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='calcium_bflow_analysis',
    version='0.1.0',
    description='Scripts to analyze calcium activity and blood flow dynamics',  # Optional
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/PBLab/python-ca-analysis-bloodflow',  # Optional
    author='PBLab/Hagai Har-Gil',  # Optional
    author_email='hagaihargil@mail.tau.ac.il',  # Optional
    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='calcium blooflow neuroscience',  # Optional
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
    install_requires=['matplotlib >= 3',
                      'numpy >= 1.18',
                      'scipy >= 1.4',
                      'tifffile',
                      'ipython >= 7',
                      'pandas >= 0.25',
                      'xarray >= 0.14',
                      'seaborn >= 0.9',
                      'ansimarkup >= 1.4',
                      'scikit-learn >= 0.22',
                      'scikit-image >= 0.16',
                      'jupyter >= 1',],

    package_data={  # Optional
        'peakutils': ['lucashnegri-peakutils-48c329c177b1'],
    },
)