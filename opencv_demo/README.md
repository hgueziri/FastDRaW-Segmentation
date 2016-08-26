# FastDRaW - OpenCV demo

This is a quick demo using OpenCV-based window for image labeling 

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

`python opencv_example.py image.png`



## List of commands:

Mouse:
   - Left click: draw foreground labels
   - Right click: draw background labels

Keyboard:
   - S: save the binary segmentation mask (in saved/)
   - L: save labelled image (in saved/)
   - C: save contour result image (in saved/)
   - Q: Quit the application
   - Z: Undo label
   - R: Switch between **Random Walker** segmentation and **FastDRaW** segmentation
   - Up arrow: increase the brush thickness
   - Down arrow: decrease the brush thickness
