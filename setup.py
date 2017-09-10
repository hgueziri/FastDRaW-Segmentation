from distutils.core import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(
  name = 'FastDRaW', 
  version = '1.2.1',
  packages = ['FastDRaW'],
  description = 'Image segmentation algorithm using Fast Delineation by RAndom Walker',
  long_description=readme(),
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
  install_requires=['markdown',],
  include_package_data = True,
  zip_safe = False
)