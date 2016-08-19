#!/usr/bin/python
"""
Created on Tue Apr 29 13:09:45 2016

FastDRaW - Fast Delineation by Random Walker

This software is under the MIT License. If you use this code in your research please
cite the following article:

H.-E. Gueziri, L. Lakhdar, M. J. McGuffin and C. Laporte "FastDRaW - Fast Delineation by Random
Walker: application to large images", Interactive Medical Image Computation (IMIC) Workshop, MICCAI 2016.

@author: Houssem-Eddine Gueziri
@contact: houssem-eddine.gueziri.1@etsmtl.net

---------------

LICENSE 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import cv2,os,time,sys
from numpy import (array,zeros,zeros_like,ones_like,hstack,vstack,dstack,arange,
                   exp,abs,ravel,ravel_multi_index,r_,where,delete,unique,empty,unravel_index,
                   __version__)
from numpy import bool as npbool
from numpy import int8 as npint8
from numpy import uint8 as npuint8
from numpy import float32 as npfloat32

from scipy.ndimage import distance_transform_edt,imread
from skimage.transform import resize as skresize
from skimage.measure import label as find_label
from skimage.morphology import dilation,disk
from skimage.segmentation import random_walker

from scipy.sparse import csr_matrix,coo_matrix
from scipy.sparse.linalg import cg as cg_solver
from threading import Thread,active_count

###############################################################################
#                   utilities to accelerate execution
###############################################################################
nparray=array
npzeros=zeros
npzeros_like=zeros_like
npones_like=ones_like
nphstack=hstack
npvstack=vstack
npdstack=dstack
npexp=exp
npabs=abs
npravel=ravel
npravel_multi_index=ravel_multi_index
npr_=r_
npwhere=where
npdelete=delete
npunique=unique
nparange=arange
npempty=empty
npunravel_index=unravel_index
skimage_version=np.float32(__version__[:4])



class Segmentation():
    
    def __init__(self,imageName):
        self.drawingR = False 
        self.drawingL = False
        self.brushSize=3
        
        self.update_cpt=1
        self.m_saveID=0
        self.maskOfInterest=None
        self.im_maskOfInterest=None
        self.img_contour=None
        self.small_img_contour=None
        self.im_to_save=None
        
        self.currentSegmentationApproach=self.FastDRaW
        
        self.RUNNING_THREADS = active_count()
        
        
        self.img = cv2.imread(imageName,cv2.CV_LOAD_IMAGE_COLOR)
        self.img_labelled=self.img.copy()
        self.markers=npzeros((self.img.shape[0],self.img.shape[1]))
        #------
        cv2.namedWindow('image')
        cv2.moveWindow('image', 0, 0)
        cv2.setMouseCallback('image',self.drawing,[self.markers])
        cv2.imshow('image',self.img)
        
        
        self.m_cpt=0
        self.m_index_array=nparray([0,0])
        
        self.RW_BETA=300
        
        self.rgbImage=imread(imageName)/255.0
        if self.rgbImage.ndim == 2:
            self.rgbImage=npdstack((z,z,z))
        elif self.rgbImage.ndim == 3:
            if self.rgbImage.shape[2]==4:
                self.rgbImage=self.rgbImage[...,0:3]
        
        self.L=self.buildGraph(self.rgbImage.mean(2),self.RW_BETA)
        self.miniL,self.im_image,self.dim = self.avt_RW(self.RW_BETA)
             
        while(1):
            
            k = cv2.waitKey(0) & 0xFF
        
            if k == ord('q'):
                break
            elif k == 82:
                self.brushSize=min(self.brushSize+1,10)
                print "brushSize",self.brushSize
            elif k == 84:
                self.brushSize=max(self.brushSize-1,0)
                print "brushSize",self.brushSize
                
         
            elif k == ord('r'):
                ## sweitch between segmentations
                if self.currentSegmentationApproach==self.FastDRaW:
                    self.currentSegmentationApproach=self.randomWalkerSegmentation
                else:
                    self.currentSegmentationApproach=self.FastDRaW
            
            elif k==ord('d'):
                cv2.imwrite("saved"+os.sep+str(self.m_saveID)+".png",self.img_labelled)
                self.m_saveID+=1
                print "Image saved - Label image"
                
            elif k==ord('s'):
                if self.img_contour is not None:
                    savedimage=self.img.copy()
                    savedimage[self.img_contour,:]=(0,255,255)
                    cv2.imwrite("saved"+os.sep+str(self.m_saveID)+".png",savedimage)
                    self.m_saveID+=1
                    print "Image saved - Contour image"
            
            elif k==ord('a'):
                if self.small_img_contour is not None:
                    cv2.imwrite("saved"+os.sep+str(self.m_saveID)+".png",(self.small_img_contour*255).astype(npuint8))
                    self.m_saveID+=1
                    print "Image saved - Downsampled contour image"
        
            elif k==ord('c'):
                if self.maskOfInterest is not None:
                    
                    # display seeds
                    ds_image=skresize(self.img,(self.dim[1],self.dim[0],3),order=0,preserve_range=True)
                    savedimage=skresize(ds_image,(self.markers.shape[0],self.markers.shape[1],3),order=0,preserve_range=True).astype(npuint8)
                    mask=npzeros_like(self.img)
                    mask[self.maskOfInterest]=(255,255,0)
                    cv2.addWeighted(savedimage, 0.7, mask, 0.3, 0.0, savedimage)
                    cv2.imwrite("saved"+os.sep+str(self.m_saveID)+".png",savedimage)
                    self.m_saveID+=1
                    print "Image saved - Ring image"
            
            elif k==ord('x'):
                    # display seeds
                ds_image=skresize(self.img,(self.dim[1],self.dim[0],3),order=0,preserve_range=True)
                savedimage=skresize(ds_image,(self.markers.shape[0],self.markers.shape[1],3),order=0,preserve_range=True).astype(npuint8)
                ROI=skresize(self.im_maskOfInterest.astype(npbool),(self.markers.shape[0],self.markers.shape[1]),order=0,preserve_range=True)
                ROI=ROI.astype(npbool)
                mask=npzeros_like(savedimage)
                mask[ROI & ~(self.markers%2==1) & ~((self.markers%2==0)&(self.markers>0))]=(255,255,0)
                cv2.addWeighted(savedimage, 0.7, mask, 0.3, 0.0, savedimage)
                cv2.imwrite("saved"+os.sep+str(self.m_saveID)+".png",savedimage)
                self.m_saveID+=1
                print "Image saved - ROI image"
        
            elif k==ord('v'):
                
                ds_image=skresize(self.img,(self.dim[1],self.dim[0],3),order=0,preserve_range=True)
                savedimage=skresize(ds_image,(self.markers.shape[0],self.markers.shape[1],3),order=0,preserve_range=True).astype(npuint8)
                cv2.imwrite("saved"+os.sep+str(self.m_saveID)+".png",savedimage)
                self.m_saveID+=1
                print "Image saved - Downsampled image"
                
            elif k == ord('z'):
                saveDrawnCont=self.markers==-1
                self.maskOfInterest=None
                if self.m_cpt>0:
                    indx=npwhere(self.m_index_array[1:,1]==self.markers.max())[0]
                    self.m_index_array=npdelete(self.m_index_array,indx+1,axis=0)
                    self.img = cv2.imread(imageName,cv2.CV_LOAD_IMAGE_COLOR)
                    self.markers=npzeros_like(self.markers)
                    for i in npunique(self.m_index_array[1:,1]):
                        indx=npwhere(self.m_index_array[:,1]==i)[0]
                        x,y=npunravel_index(self.m_index_array[indx,0],self.markers.shape)           
                        self.markers[x,y]=self.m_index_array[indx,1]
                    
                        if i%2==1:
                            self.img[x,y]=(0,0,255)
                        elif i%2==0:
                            self.img[x,y]=(0,255,0)
        
                    self.markers[saveDrawnCont]=-1
                    self.img[saveDrawnCont]=(0,255,255)
                    self.img_labelled=self.img.copy()
                    self.m_cpt-=2
                    cv2.imshow('image',self.img)
                elif any(saveDrawnCont):
                    self.markers=npzeros_like(self.markers)
                    self.img = cv2.imread(imageName,cv2.CV_LOAD_IMAGE_COLOR)
                    self.img_labelled=self.img.copy()
                    cv2.imshow('image',self.img)
                    
                
        
        print "out"
        cv2.destroyAllWindows()

    def buildGraph(self,image,RW_BETA):
        ## Building the graph: vertices, edges and weights
        vertices = nparange(image.shape[0] * image.shape[1]).reshape(image.shape)
        edges_right = npvstack((vertices[:, :-1].ravel(),vertices[:, 1:].ravel()))
        edges_down = npvstack((vertices[:-1].ravel(), vertices[1:].ravel()))
        edges = nphstack((edges_right, edges_down))
        gr_right = npabs(image[:, :-1] - image[:, 1:]).ravel()
        gr_down = npabs(image[:-1] - image[1:]).ravel()
        weights = npexp(- RW_BETA * npr_[gr_right, gr_down]**2)+1e-6
        
        ## Compute the graph's Laplacian L
        pixel_nb = edges.max() + 1
        diag = nparange(pixel_nb)
        i_indices = nphstack((edges[0], edges[1]))
        j_indices = nphstack((edges[1], edges[0]))
        data = nphstack((-weights, -weights))
        lap = coo_matrix((data, (i_indices, j_indices)),
                                shape=(pixel_nb, pixel_nb))
        connect = - npravel(lap.sum(axis=1))
        lap = coo_matrix((nphstack((data, connect)),
                         (nphstack((i_indices, diag)),
                          nphstack((j_indices, diag)))),
                          shape=(pixel_nb, pixel_nb))
        L=lap.tocsr()
        
        return L
        
    
    # mouse callback function
    def drawing(self,event,x,y,flags,param):
        
    
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawingL = True
            self.drawingR=False
            
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawingR = True
            self.drawingL=False
    
    
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawingL == True:
                self.update_cpt+=1
                if self.update_cpt%5==0:
                    self.update_cpt=1
    
                    if active_count()<=self.RUNNING_THREADS:
                        thread = Thread(target = self.currentSegmentationApproach, args = (self.markers,))
                        thread.start()
    
                cv2.circle(self.img_labelled,(x,y),self.brushSize,(0,0,255),-1)
                cv2.circle(self.markers,(x,y),self.brushSize,self.m_cpt+1,-1)
                                
            elif self.drawingR == True:
                self.update_cpt+=1
                if self.update_cpt%5==0:
                    self.update_cpt=1
                    if active_count()<=self.RUNNING_THREADS:
                        thread = Thread(target = self.currentSegmentationApproach, args = (self.markers,))
                        thread.start()
                                        
                cv2.circle(self.img_labelled,(x,y),self.brushSize,(0,255,0),-1)
                cv2.circle(self.markers,(x,y),self.brushSize,self.m_cpt+2,-1)
     
     
    
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawingL = False
            cv2.circle(self.img_labelled,(x,y),self.brushSize,(0,0,255),-1)
            cv2.circle(self.markers,(x,y),self.brushSize,self.m_cpt+1,-1)
            ixy=npravel_multi_index(npwhere(self.markers==self.m_cpt+1),self.markers.shape)
            vect=npvstack((ixy,nparray([self.m_cpt+1]*len(ixy)))).T
            self.m_index_array=npvstack((self.m_index_array,vect))
            self.m_cpt+=2
            
            while active_count()>self.RUNNING_THREADS:
                time.sleep(0.001)
            if active_count()<=self.RUNNING_THREADS:
                thread = Thread(target = self.currentSegmentationApproach, args = (self.markers,))
                thread.start()
    
        elif event == cv2.EVENT_RBUTTONUP:
            self.drawingR = False
            cv2.circle(self.img_labelled,(x,y),self.brushSize,(0,255,0),-1)
            cv2.circle(self.markers,(x,y),self.brushSize,self.m_cpt+2,-1)
            ixy=npravel_multi_index(npwhere(self.markers==self.m_cpt+2),self.markers.shape)
            vect=npvstack((ixy,nparray([self.m_cpt+2]*len(ixy)))).T
            self.m_index_array=npvstack((self.m_index_array,vect))
            self.m_cpt+=2
            
            while active_count()>self.RUNNING_THREADS:
                time.sleep(0.001)
            if active_count()<=self.RUNNING_THREADS:
                thread = Thread(target = self.currentSegmentationApproach, args = (self.markers,))
                thread.start()
    
        self.img=self.img_labelled.copy()
        if self.img_contour is not None:
            self.img[self.img_contour,:]=(0,255,255)
        cv2.imshow('image',self.img)
        
      
    
    
    
    def avt_RW(self,RW_BETA):
        image=self.rgbImage
        ratio=float(image.shape[1])/image.shape[0]
        dim=(int(100*ratio),100)
        im_image=skresize(image.mean(2),dim[::-1],order=0,preserve_range=True)
     
        miniL=self.buildGraph(im_image,RW_BETA)
        
        return miniL,im_image,dim
        
    
    
    def FastDRaW(self,markers):
        tt=time.clock()
        
        im_markers=skresize(markers,self.dim[::-1],order=0,preserve_range=True)
        im_entropyMap=0
        
        for i in range(2):
            M=npones_like(im_markers)
            M[(im_markers%2==i)&(im_markers>0)]=0
            distMap=distance_transform_edt(M)
            im_entropyMap +=  distMap
            
        im_entropyMap /=im_entropyMap.max()
        
        if self.im_maskOfInterest is None:
            self.im_maskOfInterest=(im_entropyMap<=im_entropyMap.mean() - im_entropyMap.std())
        else:
            self.im_maskOfInterest=self.im_maskOfInterest | (im_entropyMap<=im_entropyMap.mean() - im_entropyMap.std())
    
        unlabeled=npravel_multi_index(npwhere((im_markers==0)&(self.im_maskOfInterest)),im_markers.shape)
        labeled=npravel_multi_index(npwhere((im_markers>0)&(self.im_maskOfInterest)),im_markers.shape)
    
        ## Preparing the right handside of the equation BT xs
        B = self.miniL[unlabeled][:, labeled]
        mask= im_markers.flatten()[labeled]%2==1
        fs = csr_matrix(mask)
        fs = fs.transpose()
        rhs=B * fs
        
        ## Preparing the left handside of the equation Lu
        Lu=self.miniL[unlabeled][:, unlabeled]
        
    
        ## Solve the linear equation Lu xu = -BT xs
        probability=cg_solver(Lu, -rhs.todense(), tol=1e-3, maxiter=120)[0]

        miniProba=npzeros_like(im_markers,dtype=npfloat32)
        miniProba[(im_markers==0)&(self.im_maskOfInterest)]=probability
        miniProba[(im_markers%2==1)&(self.im_maskOfInterest)]=1   
    
        mask=miniProba>=0.5
        mask=(dilation(mask,disk(1))-mask).astype(npbool)
        entropyMap=distance_transform_edt(mask!=1)
        
        
#        initialGuess=skresize(miniProba,(markers.shape[0],markers.shape[1]),order=1,preserve_range=True)
        self.small_img_contour=self.rgbImage.copy()
        cont=skresize(mask,(markers.shape[0],markers.shape[1]),order=0,preserve_range=True)
        cont=cont.astype(npbool)
        self.small_img_contour[cont]=(0,1,1)
        
        self.maskOfInterest=(entropyMap<=3)
            
        labeledImage=find_label(self.maskOfInterest,background=True)
#        if skimage_version<0.12:
#            labeledImage=labeledImage-1
#            print "substraction"
        labeledImage=skresize(labeledImage,(markers.shape[0],markers.shape[1]),order=0,preserve_range=True)
        labeledImage=labeledImage.astype(npint8)
        
        self.maskOfInterest=skresize(self.maskOfInterest,(markers.shape[0],markers.shape[1]),order=0,preserve_range=True)
        self.maskOfInterest=self.maskOfInterest.astype(npbool)
        
        
        ## Extract labelled and unlabelled vertices
        m_unlabeled=(markers==0)&(self.maskOfInterest)
        m_foreground=((markers%2)==1)|(labeledImage>=1)
    
        
        unlabeled=npravel_multi_index(npwhere(m_unlabeled),markers.shape)
        labeled=npravel_multi_index(npwhere((markers>0)|(labeledImage>=0)),markers.shape)
        
        ## Preparing the right handside of the equation BT xs
        B = self.L[unlabeled][:, labeled]
        mask= (m_foreground).flatten()[labeled]
        fs = csr_matrix(mask).transpose()
        rhs=B * fs
        
        ## Preparing the left handside of the equation Lu
        Lu=self.L[unlabeled][:, unlabeled]
        
        ## Solve the linear equation Lu xu = -BT xs
        probability=cg_solver(Lu, -rhs.todense(),x0=(unlabeled*0)+0.5,tol=1e-3, maxiter=120)[0]
        
        
        ## Display the result at 0.5 probability threshold   
        x0=npzeros_like(markers,dtype=npfloat32)
        x0[m_unlabeled]=probability
        x0[m_foreground]=1
        
        
        mask=x0>=0.5
        mask=(dilation(mask,disk(2))-mask).astype(npbool)
        self.img_contour=mask.copy()
        self.img=self.img_labelled.copy()
        self.img[mask,:]=(0,255,255)
        cv2.imshow('image',self.img)
        
        print "------------------------------ FastDRaW:",time.clock()-tt
        
        
        
    def randomWalkerSegmentation(self,markers):
        tt=time.clock()
        
        image=self.rgbImage
        markersBin=npzeros_like(markers)
        markersBin[markers%2==1]=1
        markersBin[markers%2==0]=2
        markersBin[markers==0]=0
    
        probability=random_walker(image, markersBin, beta=self.RW_BETA, mode='cg_mg',
                                  return_full_prob=True,multichannel=True)
    
        ## Display the result at 0.5 probability threshold
        mask=probability[0,...]>=0.5
        mask=dilation(mask,disk(2))-mask
        self.img_contour=mask.copy()
        self.img=self.img_labelled.copy()
        self.img[mask,:]=(0,255,255)
        cv2.imshow('image',self.img)
        
        print "------------------------------ RW:",time.clock()-tt
        

    
    
    
###############################################################################
#                                   MAIN                                      #
###############################################################################

if __name__=="__main__":

    if len(sys.argv)<2:
	imageName="image.png"
    else:
	imageName=sys.argv[1]
    Segmentation(imageName)

