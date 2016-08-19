# FastDRaW
FastDRaW – Fast Delineation by Random

This software is under the MIT License. If you use this code in your research please cite the following paper:

H.-E. Gueziri, L. Lakhdar, M. J. McGuffin and C. Laporte "FastDRaW – Fast Delineation by Random Walker: application to large images", MICCAI workshop on Interactive Medical Image Computation (IMIC), Athens, Greece, (2016).

@author Houssem-Eddine Gueziri

@contact houssem-eddine.gueziri.1@etsmtl.net

## Requirements:

To run the demo you should install the following dependencies

- [Scipy and Numpy](https://www.scipy.org/install.html)
- [PyOpenCV](https://pypi.python.org/pypi/pyopencv/2.1.0.wr1.2.0) (for window management)
- [scikit-image](http://scikit-image.org/docs/dev/install.html) (for image processing tools)
- [PyAMG](http://pyamg.org/) (for linear system solver)

Or run

```
apt-get install build-essential python2.7-dev python-opencv
pip install -r requirements.txt
```


## Example:

`python FastDRaW.py image.png`



## Command list:

Mouse:
   - Left click: draw foreground labels
   - Right click: draw background labels

Keyboard:
   - A: save the current down-sampled segmentation result (in saved/)
   - S: save the current full-resolution segmentation result (in saved/)
   - D: save labelled image (in saved/)
   - C: save the refinement strip image (in saved/)
   - X: save the ROI image (in saved/)
   - Q: Quit the application
   - Z: Undo label
   - R: Launch RW segmentation
   - G: Launch GRISLI segmentation
   - Up arrow: increase the brush thickness
   - Down arrow: decrease the brush thickness
