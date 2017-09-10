from distutils.core import setup

setup(
  name = 'FastDRaW', 
  version = '1.2.0',
  packages = ['FastDRaW'],
  description = 'Image segmentation algorithm using Fast Delineation by RAndom Walker',
  license = 'MIT',
  author = 'Houssem-Eddine Gueziri',
  author_email = 'houssem-eddine.gueziri.1@etsmtl.net',
  url = 'https://github.com/hgueziri/FastDRaW-Segmentation', 
  download_url = 'https://github.com/hgueziri/FastDRaW-Segmentation/archive/1.2.tar.gz', 
  keywords = ['segmentation', 'fastdraw', 'medical'], 
  classifiers = ['Development Status :: 4 - Beta',
  				 'Intended Audience :: Developers',
    			 'License :: OSI Approved :: MIT License',
    			 'Programming Language :: Python :: 2.7'],
)