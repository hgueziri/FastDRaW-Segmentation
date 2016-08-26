#!/usr/bin/python
"""
Created on Tue Aug 26

This is an OpenCV-based demo for FastDRaW

Please refer to the README.md file for details

@author: Houssem-Eddine Gueziri

"""

import cv2,os,time,sys
import numpy as np


from skimage import img_as_float
from skimage.color import rgb2grey

from scipy.ndimage import distance_transform_edt,imread
from skimage.transform import resize
from skimage.measure import label as find_label
from skimage.morphology import dilation,disk
from skimage.segmentation import random_walker

from scipy.sparse import csr_matrix,coo_matrix
from scipy.sparse.linalg import cg as cg_solver
from threading import Thread,active_count

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
skimage_version=np.float32(np.__version__[:4])
from FastDRaW import FastDRaW


class InteractiveSegmentation():
    """This class handles the window event (mouse and keyboard).

    The constructor runs an endless for loop that waits for keyboard events.
    Keyboard keys are detected with a simple, but long, 'if ... elif ... else'
    condition in the for loop. Mouse events are detected via callback function
    that are implemented as `drawing` method. FastDRaW and random walker
    segmentations are launched in a separated thread using `run_fastDRaW` and
    `run_random_walker` methods, respectively.
    """

    def __init__(self,imageName):
        # initialize variables for mouse interaction
        self.drawingR = False # right click pressed
        self.drawingL = False #left click pressed
        self.brushSize = 3 # size of the drawing
        self.m_saveID = 0 # id to save images
        # counter to enable undo actions
        self.update_cpt = 1
        self.m_cpt = 0
        self.m_index_array = np.array([0, 0])

        # initialize beta parameter for RW
        self.RW_BETA = 300

        # initialize variables for display
        self.img_contour = None # rgb image with contour result
        self.mask_segmentation = None # binary segmentation result

        # pointer to the segmentation method (default FastDRaW)
        self.selectedApproach = self.run_fastDRaW

        # initialize the number of thread running
        self.RUNNING_THREADS = active_count()

        # read the image `self.img` serves as original image and
        # is not modified
        self.img = cv2.imread(imageName, cv2.CV_LOAD_IMAGE_COLOR)
        # current image to display `self.img_labelled` labels are drawn on
        # this image
        self.img_labelled = self.img.copy()
        # array of labels (int)
        self.markers = np.zeros((self.img.shape[0], self.img.shape[1]))
        # gray-level image to be processed
        self.image = img_as_float(rgb2grey(self.img))

        # display the image and set callback to detect mouse events
        cv2.namedWindow('image')
        cv2.moveWindow('image', 0, 0)
        cv2.setMouseCallback('image', self.drawing, [])
        cv2.imshow('image', self.img)

        # instanciate FastDRaW object:
        #   initialize
        self.fastdraw = FastDRaW(self.image, # gray-level 2D image
                                 beta=self.RW_BETA, # beta to build the graph
                                 downsampled_size=100, # size of coarse image
                                 tol=1.e-3, # tolerence for solving RW equation
                                 return_full_prob=False # if True returns
                                                        # probabilities instead
                                                        # of binary image
                                 )

        # opencv main loop
        while(1):

            # wait for keyboard press
            k = cv2.waitKey(0) & 0xFF

            if k == ord('q'):
                # if 'q' pressed leave the loop and close windows
                break
            elif k == 82:
                # if up arrow is pressed, increase brush size
                self.brushSize = min(self.brushSize + 1, 10)
                print "brushSize",self.brushSize
            elif k == 84:
                # if down arrow is pressed, decrease brush size
                self.brushSize = max(self.brushSize - 1, 0)
                print "brushSize",self.brushSize


            elif k == ord('r'):
                # sweitch between segmentations
                if self.selectedApproach==self.run_fastDRaW:
                    self.selectedApproach=self.run_random_walker
                else:
                    self.selectedApproach=self.run_fastDRaW

            elif k == ord('l'):
                # save labeled image
                cv2.imwrite("saved" + os.sep + str(self.m_saveID) + ".png",
                            self.img_labelled)
                self.m_saveID = self.m_saveID + 1
                print "Image saved - Label image"

            elif k == ord('c'):
                # save contour image
                if self.img_contour is not None:
                    savedimage = self.img.copy()
                    savedimage[self.img_contour, :]=(0, 255, 255)
                    cv2.imwrite("saved" + os.sep + str(self.m_saveID) + ".png",
                                savedimage)
                    self.m_saveID = self.m_saveID + 1
                    print "Image saved - Contour image"

            elif k == ord('s'):
                # save binary segmentation image
                if self.mask_segmentation is not None:
                    savedimage = np.uint8(self.mask_segmentation * 255)
                    cv2.imwrite("saved" + os.sep + str(self.m_saveID) + ".png",
                                savedimage)
                    self.m_saveID = self.m_saveID + 1
                    print "Image saved - Segmentation result image"

            elif k == ord('z'):
                # if 'z' is pressed, find and delete the last label drown
                # `m_cpt`:
                #   foreground labels are odd
                #   background labels are even
                #   find the largest number in `markers` array and remove
                saveDrawnCont = self.markers == -1
                self.maskOfInterest = None
                if (self.m_cpt > 0):
                    indx = np.where(self.m_index_array[1:, 1] == \
                                    self.markers.max())[0]
                    self.m_index_array = np.delete(self.m_index_array, indx+1,
                                                   axis=0)
                    self.img = cv2.imread(imageName, cv2.CV_LOAD_IMAGE_COLOR)
                    self.markers = np.zeros_like(self.markers)
                    for i in np.unique(self.m_index_array[1:, 1]):
                        indx = np.where(self.m_index_array[:, 1] == i)[0]
                        x,y = np.unravel_index(self.m_index_array[indx, 0],
                                               self.markers.shape)
                        self.markers[x,y] = self.m_index_array[indx, 1]

                        if (i%2) == 1:
                            self.img[x, y] = (0, 0, 255)
                        elif (i%2) == 0:
                            self.img[x, y] = (0, 255, 0)

                    self.markers[saveDrawnCont] = -1
                    self.img[saveDrawnCont] = (0, 255, 255)
                    self.img_labelled = self.img.copy()
                    self.m_cpt = self.m_cpt - 2
                    cv2.imshow('image', self.img)

                elif any(saveDrawnCont):
                    self.markers = np.zeros_like(self.markers)
                    self.img = cv2.imread(imageName, cv2.CV_LOAD_IMAGE_COLOR)
                    self.img_labelled = self.img.copy()
                    cv2.imshow('image', self.img)



        print "out"
        cv2.destroyAllWindows()



    # mouse callback function
    def drawing(self, event, x, y, flags, param):
        """This function is called when a mouse event is detected"""

        if event == cv2.EVENT_LBUTTONDOWN:
            # left click
            self.drawingL = True
            self.drawingR = False


        elif event == cv2.EVENT_RBUTTONDOWN:
            # right click
            self.drawingR = True
            self.drawingL = False


        elif event == cv2.EVENT_MOUSEMOVE:
            # mouse is moving
            if self.drawingL == True:
                # if left click is pressed when mouse is moving over the image
                # draw red on the image to display
                cv2.circle(self.img_labelled, (x, y), self.brushSize,
                           (0, 0, 255), -1)
                # draw odd values on the label image
                cv2.circle(self.markers, (x, y), self.brushSize,
                           self.m_cpt + 1, -1)
                # update the segmentation every 5 mouvement in `update_cpt`
                self.update_cpt += 1
                if self.update_cpt%5 == 0:
                    self.update_cpt = 1
                    if active_count() <= self.RUNNING_THREADS:
                        # if no segmentation is currently running
                        # run the segmentation on a separate thread
                        thread = Thread(target=self.selectedApproach,
                                        args=(self.markers,))
                        thread.start()

            elif self.drawingR == True:
                # if right click is pressed when mouse is moving over the image
                # draw green on the image to display
                cv2.circle(self.img_labelled, (x, y), self.brushSize,
                           (0, 255, 0), -1)
                # draw even values on the label image
                cv2.circle(self.markers, (x, y), self.brushSize,
                           self.m_cpt + 2, -1)
                # update the segmentation every 5 mouvement in `update_cpt`
                self.update_cpt += 1
                if self.update_cpt%5 == 0:
                    self.update_cpt = 1
                    if active_count() <= self.RUNNING_THREADS:
                        # if no segmentation is currently running
                        # run the segmentation on a separate thread
                        thread = Thread(target=self.selectedApproach,
                                        args=(self.markers,))
                        thread.start()





        elif event == cv2.EVENT_LBUTTONUP:
            # on left button release
            self.drawingL = False
            # draw the last position of the mouse on the image and label image
            cv2.circle(self.img_labelled, (x, y), self.brushSize,
                       (0, 0, 255), -1)
            cv2.circle(self.markers, (x, y), self.brushSize,
                       self.m_cpt + 1, -1)
            # increase the counter for labels by 2 to enable undo
            # odd for foreground
            # even for background
            ixy = np.ravel_multi_index(np.where(self.markers == self.m_cpt+1),
                                       self.markers.shape)
            vect = np.vstack((ixy, np.array([self.m_cpt+1] * len(ixy)))).T
            self.m_index_array = np.vstack((self.m_index_array, vect))
            self.m_cpt = self.m_cpt + 2

            # if a segmentation is currently running, wait
            while active_count() > self.RUNNING_THREADS:
                time.sleep(0.001)
            # run the segmentation
            if active_count() <= self.RUNNING_THREADS:
                thread = Thread(target=self.selectedApproach,
                                args=(self.markers,))
                thread.start()

        elif event == cv2.EVENT_RBUTTONUP:
            self.drawingR = False
            cv2.circle(self.img_labelled, (x, y), self.brushSize,
                       (0, 255, 0), -1)
            cv2.circle(self.markers, (x, y), self.brushSize,
                       self.m_cpt + 2, -1)
            # increase the counter for labels by 2 to enable undo
            # odd for foreground
            # even for background
            ixy = np.ravel_multi_index(np.where(self.markers == self.m_cpt+2),
                                       self.markers.shape)
            vect = np.vstack((ixy, np.array([self.m_cpt+2] * len(ixy)))).T
            self.m_index_array = np.vstack((self.m_index_array, vect))
            self.m_cpt = self.m_cpt + 2

            # if a segmentation is currently running, wait
            while active_count() > self.RUNNING_THREADS:
                time.sleep(0.001)
            # run the segmentation
            if active_count() <= self.RUNNING_THREADS:
                thread = Thread(target=self.selectedApproach,
                                args=(self.markers,))
                thread.start()

        # refresh displayed image
        self.img = self.img_labelled.copy()
        if self.img_contour is not None:
            self.img[self.img_contour, :]=(0, 255, 255)
        cv2.imshow('image', self.img)




    def run_fastDRaW(self, markers):
        tt = time.clock() # to measure the elapsed time

        # `markers` array contains foreground and background labels under odd
        # and even numbers, respectively. Here `markers` is converted to
        # contain 1 for foreground, 2 for background labels and zero for
        # unlabeled pixels
        bin_markers = np.zeros_like(markers)
        bin_markers[(markers%2 == 1)] = 1
        bin_markers[(markers%2 == 0) & (markers > 0)] = 2

        # run FastDRaW segmentation
        # `bin_markers` is the converted markers array
        # `k` is a parameter to control the ROI (large k for large ROI)
        # small ROI increase the segmentation speed
        # too small ROI cause segmentation error, because the object could
        #   not be entirely inside the ROI
        # a too large ROI would contain the entire down-sampled image
        mask = self.fastdraw.update(bin_markers, k=1)

        # copy the result for save purposes (when 's' key is pressed)
        self.mask_segmentation = mask.copy()

        # compute the contour of the object
        mask = (dilation(mask, disk(2)) - mask).astype(np.bool)

        # copy image for display purposes
        self.img_contour = mask.copy()
        self.img = self.img_labelled.copy()
        self.img[mask, :] = (0, 255, 255)
        cv2.imshow('image', self.img)

        # print the elapsed time
        print "------------------------------ FastDRaW:",time.clock()-tt



    def run_random_walker(self, markers):
        tt = time.clock() # to measure the elapsed time

        # `markers` array contains foreground and background labels under odd
        # and even numbers, respectively. Here `markers` is converted to
        # contain 1 for foreground, 2 for background labels and zero for
        # unlabeled pixels
        markersBin = np.zeros_like(markers)
        markersBin[markers%2 == 1] = 1
        markersBin[markers%2 == 0] = 2
        markersBin[markers == 0] = 0

        # run random walker segmentation
        # (see http://scikit-image.org/ for details)
        result = random_walker(self.image, markersBin, beta=self.RW_BETA,
                                    mode='cg_mg',return_full_prob=False,
                                    multichannel=True)

        # convert to binary image by selecting only one label
        mask = result == 1

        # copy the result for save purposes (when 's' key is pressed)
        self.mask_segmentation = mask.copy()

        # compute the contour of the object
        mask = dilation(mask, disk(2)) - mask

        # copy image for display purposes
        self.img_contour = mask.copy()
        self.img = self.img_labelled.copy()
        self.img[mask, :] = (0, 255, 255)
        cv2.imshow('image', self.img)

        # print the elapsed time
        print "------------------------------ RW:", time.clock() - tt

"""Here we write the main script to run the application"""

if __name__=="__main__":

    if len(sys.argv) < 2:
        # if no argument is given as image name, e.g.,
        # python opencv_example.py
        imageName = "image.png"
    else:
        # if argument is given as image name, e.g.,
        # python opencv_example.py image.png
        imageName = sys.argv[1]
    # run the application
    InteractiveSegmentation(imageName)

