# FastDRaW
FastDRaW – Fast Delineation by Random

This software is under the MIT License. If you use this code in your research please cite the following paper:

H.-E. Gueziri, L. Lakhdar, M. J. McGuffin and C. Laporte "FastDRaW – Fast Delineation by Random Walker: application to large images", MICCAI workshop on Interactive Medical Image Computation (IMIC), Athens, Greece, (2016).

@author Houssem-Eddine Gueziri

## Requirements:

FastDRaW requires the following packages

- [Scipy and Numpy](https://www.scipy.org/install.html)
- [scikit-image](http://scikit-image.org/docs/dev/install.html) (for image processing tools)
- [PyAMG](http://pyamg.org/) (for linear system solver)

Or run

```shell
apt-get install build-essential python2.7-dev
pip install -r requirements.txt
```

## Example:

```python
>>> from skimage.data import coins
>>> import matplotlib.pyplot as plt
>>> image = coins()
>>> labels = np.zeros_like(image)
>>> labels[[129, 199], [155, 155]] = 1 # label some pixels as foreground
>>> labels[[162, 224], [131, 184]] = 2 # label some pixels as background
>>> fastdraw = FastDRaW(image, beta=100, downsampled_size=100)
>>> segm = fastdraw.update(labels)
>>> plt.imshow(image,'gray')
>>> plt.imshow(segm, alpha=0.7)
```

An example using matplotlib for interactive demonstration is provided in 'plot_fast_draw_segmentation.py'. Use the mouse to draw labels, left click for foreground and right click for background.
Use 'space bar' key to perform a segmentation.

```shell
python plot_fast_draw_segmentation.py
```

A more advanced example using OpenCV interaction window is provided in 'opencv_demo/opencv_example.py' (requires to install PyOpenCV, see opencv_demo/README.md file for more details)

```shell
cd  opencv_demo
python opencv_example.py image.png
```

