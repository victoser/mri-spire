from setuptools import setup, find_packages

setup(
  name = 'mri-spire',
  version = '0.1',      
  license='MIT',
  description = 'Sparse Iterative Reconstructor for quantitative MRI',
  author = 'Victor Serban', 
  author_email = 'serban.victor@gmail.com', 
  url = 'https://github.com/victoser/mri-spire', 
  download_url = 'https://github.com/victoser/mri-spire/archive/v0.1.tar.gz', 
  packages = find_packages(), 
  install_requires=[  
          'numpy',
          'matplotlib',
          'scipy'
  ],
  classifiers=[
    'Intended Audience :: Researchers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)