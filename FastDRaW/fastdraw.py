from scipy.ndimage import distance_transform_edt
import numpy as np

from scipy import sparse
from scipy.sparse.linalg import cg

from scipy.ndimage import zoom
from scipy.ndimage import generate_binary_structure
from scipy.ndimage.morphology import binary_dilation

from warnings import warn

class Segmenter():
    """A fast segmentation algorithm based on the random walker algorithm.
        
    FastDRaW implemented for 2D images as described in [1].
    The algorithm performs in a two-step segmetnation. In the first step, a
    random walker segmentation is performed on a small (down-sampled)
    version of the image to obtain a coarse segmentation contour. In the
    second step, the result is refined by applying a second random walker
    segmentation over a narrow strip around the coarse contour.
    
    Parameters
    ----------
    image : array_like
        Image to be segmented. If `image` is multi-channel it will
        be converted to gray-level image before segmentation.
    beta : float
        Penalization coefficient for the random walker motion
        (the greater `beta`, the more difficult the diffusion).
    downsampled_size : int, default 100
        The size of the down-sampled image. Should be smaller than the
        size of the original image. Recommended values between 100 and 200.
    tol : float
        tolerance to achieve when solving the linear system, in
        cg' and 'cg_mg' modes.
    return_full_prob : bool, default False
        If True, the probability that a pixel belongs to each of the labels
        will be returned, instead of boolean array.
    
    See also
    --------
    skimage.segmentation.random_walker: random walker segmentation
        The original random walker algorithm.
        
    References
    ----------
    [1] H.-E. Gueziri, L. Lakhdar, M. J. McGuffin and C. Laporte,
    "FastDRaW - Fast Delineation by Random Walker: application to large
    images", MICCAI Workshop on Interactive Medical Image Computing (IMIC),
    Athens, Greece, (2016).
    
    Examples
    --------
    >>> from skimage.data import coins
    >>> import matplotlib.pyplot as plt
    >>> image = coins()
    >>> labels = np.zeros_like(image)
    >>> labels[[129, 199], [155, 155]] = 1 # label some pixels as foreground
    >>> labels[[162, 224], [131, 184]] = 2 # label some pixels as background
    >>> fastdraw = Segmenter(image, beta=100, downsampled_size=100)
    >>> segm = fastdraw.update(labels)
    >>> plt.imshow(image,'gray')
    >>> plt.imshow(segm, alpha=0.7)
    """
    
    def __init__(self, image, beta=300, downsampled_size=[100,100], tol=1.e-3):
        
        try:
            from pyamg import ruge_stuben_solver
            self._pyamg_found = True
        except ImportError:
            self._pyamg_found = False

        assert (beta > 0), 'beta should be positive.'

        assert (len(downsampled_size) == 2), 'incorrect value of downsampled_size, it should contain 2 elements: e.g., [100,100].'
        assert (downsampled_size[0] > 0), 'incorrect value, downsampled_size must be > 0.'
        assert (downsampled_size[1] > 0), 'incorrect value, downsampled_size must be > 0.'

        self._original_size = image.shape
        if image.ndim < 2 or image.ndim > 3:
            raise ValueError('Image must be of dimension 2 or 3.')
        else:
            image = np.atleast_3d(image)

        if min(downsampled_size) > min(image.shape[:-1]):
            warn('The size of the downsampled image if larger than the '
                 'original image. The computation time could be affected.')


        self.size = image.shape
        self.ds_size = (downsampled_size[0], downsampled_size[1], image.shape[2])
        
        ds_image = zoom(image, zoom=np.float32(self.ds_size)/self.size, order=2)
        ## It is important to normalize the data between [0,1] so that beta
        ## makes sens
        ds_image = np.float32(ds_image) / 255.0
        image = np.float32(image) / 255.0

        ## `full_to_ds_ratio` is used to convert labels from full resolution
        ## image to down-sampled image
        self.full_to_ds_ratio = (float(self.ds_size[0])/image.shape[0],
                                 float(self.ds_size[1])/image.shape[1])

        ## Compute the graph's Laplacian for both the full resolution `L` 
        ## and the down-sampled `ds_L` images
        self.L = self._buildGraph(image, beta)
        self.ds_L = self._buildGraph(ds_image, beta)


    def _buildGraph(self, image, RW_BETA):
        """Builds the graph: vertices, edges and weights from the `image` 
        and returns the graph's Laplacian `L`.
        """
        ## Building the graph: vertices, edges and weights
        vertices = np.arange(image.shape[0] * image.shape[1] * image.shape[2]).reshape(image.shape)
        edges_deep = np.vstack((vertices[:, :, :-1].ravel(), vertices[:, :, 1:].ravel()))
        edges_right = np.vstack((vertices[:, :-1].ravel(),vertices[:, 1:].ravel()))
        edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
        edges = np.hstack((edges_deep, edges_right, edges_down))
        gr_deep = np.abs(image[:, :, :-1] - image[:, :, 1:]).ravel() / 0.2
        gr_right = np.abs(image[:, :-1] - image[:, 1:]).ravel()
        gr_down = np.abs(image[:-1] - image[1:]).ravel()
        weights = np.exp(- RW_BETA * np.r_[gr_deep, gr_right, gr_down]**2)+1e-6
        
        ## Compute the graph's Laplacian L
        pixel_nb = edges.max() + 1
        diag = np.arange(pixel_nb)
        i_indices = np.hstack((edges[0], edges[1]))
        j_indices = np.hstack((edges[1], edges[0]))
        data = np.hstack((-weights, -weights))
        lap = sparse.coo_matrix((data, (i_indices, j_indices)),
                                shape=(pixel_nb, pixel_nb))
        connect = - np.ravel(lap.sum(axis=1))
        lap = sparse.coo_matrix((np.hstack((data, connect)),
                         (np.hstack((i_indices, diag)),
                          np.hstack((j_indices, diag)))),
            shape=(pixel_nb, pixel_nb))
        L=lap.tocsr()
        
        return L 

        
    def _check_parameters(self, labels, target_label):
        if target_label not in np.unique(labels):
            warn('The target label '+str(target_label)+ \
                 ' does not match any label')
            return 1
        if (labels != 0).all():
            warn('The segmentation is computed on the unlabeled area '
                 '(labels == 0). No zero valued areas in labels were '
                 'found. Returning provided labels.')
            return -1
        return 1
                
    def _compute_relevance_map(self, labels):
        """Computes the relevance map from labels and initialize 
        down-sampled label image `ds_labels`.
        
        The relevance map assumes that the object boundary is more likely to
        be located somwhere between different label categories, i.e. two labels
        of the same category generate a low energy, where two labels of 
        different categories generate high energy, therefore precluding 
        redundent label information. The relevance map is computed using the 
        sum of the distance transforms for each label category.
        """
        
        ds_labels = np.zeros(self.ds_size)
        ds_relevance_map = 0
        for i in np.unique(labels):
            if i != 0:
                # 2.1- Compute the coarse label image
                y,x,z = np.where(labels == i)
                ds_labels[np.int32(y*self.full_to_ds_ratio[0]),
                          np.int32(x*self.full_to_ds_ratio[1]),z] = i
                # 2.2- Compute the energy map
                M = np.ones_like(ds_labels)
                M[ds_labels == i] = 0
                distance_map = distance_transform_edt(M)
                ds_relevance_map +=  distance_map
        
        # 2.3- Normalize the energy map and compute the ROI
        ds_relevance_map = ds_relevance_map / ds_relevance_map.max()
        return ds_labels, ds_relevance_map
        
    def _coarse_random_walker(self, ds_labels, ds_maskROI, target_label):
        """Performs a coarse random walker segmentation on the down-sampled
        image.
        
        Parameters
        ----------
        target_label : int
            The label category to comput the segmentation for. `labels` should
            contain at least one pixel with value `target_label`
        
        Returns
        -------
        ds_probability : ndarray of the same size as `ds_image`
            Array of the probability between [0.0, 1.0], of each pixel
            to belong to `target_label`
        """
        unlabeled = np.ravel_multi_index(np.where((ds_labels == 0) & \
                        (ds_maskROI)), ds_labels.shape)
        labeled = np.ravel_multi_index(np.where((ds_labels > 0) & \
                        (ds_maskROI)), ds_labels.shape)
        # 3.1- Preparing the right handside of the equation BT xs
        B = self.ds_L[unlabeled][:, labeled]
        mask = ds_labels.flatten()[labeled] == target_label
        fs = sparse.csr_matrix(mask)
        fs = fs.transpose()
        rhs = B * fs
        # 3.2- Preparing the left handside of the equation Lu
        Lu = self.ds_L[unlabeled][:, unlabeled]
        # 3.3- Solve the linear equation Lu xu = -BT xs
        if self._pyamg_found:
            ml = ruge_stuben_solver(Lu)
            M = ml.aspreconditioner(cycle='V')
        else:
            M = None
        xu = cg(Lu, -rhs.todense(), tol=1e-3, M=M, maxiter=120)[0]

        ds_probability = np.zeros_like(ds_labels, dtype=np.float32)
        ds_probability[(ds_labels == 0) & (ds_maskROI)] = xu
        ds_probability[(ds_labels == target_label) & (ds_maskROI)] = 1
        
        return ds_probability
        
    def _refinement_random_walker(self, ds_labels, ds_maskROI, ds_mask, target_label):
        """Performs a random walker segmentation over a small region 
        `self.ds_maskROI` around the coarse contour on the full resolution image. 
        
        Requires `target_label` and `labels`
        
        Returns
        -------
        probability : ndarray of the same size as `image`
            Array of the probability between [0.0, 1.0], of each pixel
            to belong to `target_label`
        """

        ds_labels[(ds_maskROI==False) & ds_mask] = target_label
        ds_labels[(ds_maskROI==False) & (ds_mask==False)] = -1

        
        labels = zoom(ds_labels, zoom=np.float32(self.size)/self.ds_size, order=0)
        maskROI = zoom(ds_maskROI, zoom=np.float32(self.size)/self.ds_size, order=0).astype(np.bool)

        # Extract labelled and unlabelled vertices
        m_unlabeled = (labels == 0) & (maskROI)
        m_foreground = (labels == target_label)
        
        unlabeled = np.ravel_multi_index(np.where(m_unlabeled), self.size)
        labeled = np.ravel_multi_index(np.where(labels != 0), self.size)
        #labeled = np.ravel_multi_index(np.where((m_foreground) | (labels > 0)), self.size)

        # Preparing the right handside of the equation BT xs
        B = self.L[unlabeled][:, labeled]
        mask = (labels[labels != 0]).flatten() == target_label
        fs = sparse.csr_matrix(mask).transpose()
        rhs = B * fs
        
        # Preparing the left handside of the equation Lu
        Lu = self.L[unlabeled][:, unlabeled]
        
        # Solve the linear equation Lu xu = -BT xs
        if self._pyamg_found:
            ml = ruge_stuben_solver(Lu)
            M = ml.aspreconditioner(cycle='V')
        else:
            M = None
        xu = cg(Lu, -rhs.todense(), tol=1e-3, M=M, maxiter=120)[0]
        
        probability = np.zeros(self.size, dtype=np.float32)
        probability[m_unlabeled] = xu
        probability[m_foreground] = 1
        
        return probability
        
    def update(self, labels, target_label=1, k=0.5):
        """Updates the segmentation according to `labels` using the
        FastDRaW algorithm.
        
        The segmentation is computed in two stages. (1) coputes a coarse 
        segmentation on a down-sampled version of the image, (2) refines the 
        segmentation on the original image.
        
        Parameters
        ----------
        labels : array of ints, of same shape as `image`
            Array of seed markers labeled with different positive integers
            (each label category is represented with an integer value). 
            Zero-labeled pixels represent unlabeled pixels.
        target_label : int
            The label category to comput the segmentation for. `labels` should
            contain at least one pixel with value `target_label`
        k : float
            Control the size of the region of interest (ROI). Large positive
            value of `k` allows a larger ROI.
        
        Returns
        -------
        output : ndarray
            * If `return_full_prob` is False, array of bools of same shape as
              `image`, in which pixels have been labeled True if they belong
              to `target_label`, and False otherwise.
            * If `return_full_prob` is True, array of floats of same shape as
              `image`. in witch each pixel is assigned the probability to
              belong to `target_label`.
        """
        ## 1- Checking if inputs are valide
        labels = np.atleast_3d(labels)
        _err = self._check_parameters(labels, target_label)
        if _err == -1:
            segm = labels == target_label
            return segm
        

        ## 2- Create down-sampled (coarse) image size 
        ## and compute the energy map
        ds_labels, ds_relevance_map = self._compute_relevance_map(labels)
        
        # Threshold the energy map and append new region to the existing ROI
        threshold = ds_relevance_map.mean() + k*ds_relevance_map.std()
        ds_maskROI = ds_relevance_map <= threshold
    
        ## 3- Performe a corse RW segmentation on the down-sampled image
        ds_probability = self._coarse_random_walker(ds_labels, ds_maskROI, target_label) 
        
        # Compute the corse segmentation result 
        ds_mask = ds_probability >= 0.5
        ds_contour = np.zeros(self.ds_size, dtype=np.bool)
        ds_maskROI = np.zeros(self.ds_size, dtype=np.bool)
        struct_el2D = generate_binary_structure(2, 1)
        for i in range(self.ds_size[2]):
            ds_contour[..., i] = (binary_dilation(ds_mask[..., i], struct_el2D) - ds_mask[..., i]).astype(np.bool)
            ds_maskROI[..., i] = binary_dilation(ds_contour[..., i], struct_el2D, iterations=2)

        # Compute the refinement region around the corse result
        #maskROI = binary_dilation(ds_contour, generate_binary_structure(3,1), iterations=3)
            
        ## 4- Performe a fine RW segmentation on the full resolution image
        ##    only on the refinement region
        probability = self._refinement_random_walker(ds_labels, ds_maskROI, ds_mask, target_label)

        
        binary_segmentation = probability >= 0.5

        ## to return only contour
        # dilated_mask = np.zeros(self.size, dtype=np.bool)
        # for i in range(mask.shape[2]):
        #     dilated_mask[..., i] = binary_dilation(mask[...,i], struct_el2D).astype(np.bool)
        # mask = dilated_mask - mask

        return np.reshape(binary_segmentation, self._original_size)

     
