# FastDRaW
FastDRaW – Fast Delineation by Random

This software is under the MIT License. If you use this code in your research please cite the following paper:

H.-E. Gueziri, L. Lakhdar, M. J. McGuffin and C. Laporte "FastDRaW – Fast Delineation by Random Walker: application to large images", MICCAI workshop on Interactive Medical Image Computing (IMIC), Athens, Greece, (2016).

@author Houssem-Eddine Gueziri

## Requirements:

FastDRaW requires the following packages

- [Scipy and Numpy](https://www.scipy.org/install.html)
- [PyAMG](http://pyamg.org/) (for linear system solver)

Or run

```shell
sudo apt-get install build-essential python2.7-dev
sudo pip install -r requirements.txt
```

## Install:

From PyPI:

```shell
sudo pip install FastDRaW
```
From git repository:

```shell
git clone https://github.com/hgueziri/FastDRaW-Segmentation.git
cd FastDRaW-Segmentation
sudo python setup.py install
```

## Usage example:

```python
>>> from FastDRaW import Segmenter
>>> from skimage.data import coins
>>> import matplotlib.pyplot as plt
>>> image = coins()
>>> labels = np.zeros_like(image)
>>> labels[[129, 199], [155, 155]] = 1 # label some pixels as foreground
>>> labels[[162, 224], [131, 184]] = 2 # label some pixels as background
>>> fastdraw = Segmenter(image, beta=100, downsampled_size=[100,100])
>>> segm = fastdraw.update(labels)
>>> plt.imshow(image,'gray')
>>> plt.imshow(segm, alpha=0.7)
```


