import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
import cv2 
import scipy.signal as signal
from skimage.feature import corner_harris, peak_local_max

# *** Part 1. Defining Correspondences

# IMPORTED METHOD CODE FROM PROJECT 2
# proj2: returns a gaussian blurred img 
def gaussian(img, sigma):
    k1d = cv2.getGaussianKernel(sigma * 6, sigma)
    kernel = k1d * k1d.T
    return signal.convolve(img, np.expand_dims(kernel, axis=2), mode="same")

# proj2: creates gaussian stack starting with sigma = 2, 4, 8, 16, 32
def gauss_stack(hybrid, N):
  gstack = [hybrid]
  for i in range(1, N):
    gauss = gaussian(hybrid, 2**i)
    gstack.append(gauss) 
  return gstack

# proj2: creates lap stack of N levels
def lap_stack(gstack, N):
  lstack = []
  for i in range(0, N-1):
    lstack.append(gstack[i]-gstack[i+1]) 
    if (i == N-2):
      lstack.append(gstack[i+1])
  return lstack

# proj 4A ~~

# define n pairs of corresponding points on the two images by hand using gpinput
def get_points(im1, im2, n):
    plt.imshow(im1)
    im1_pts = np.asarray(plt.ginput(n, 0))
    plt.imshow(im2)
    im2_pts = np.asarray(plt.ginput(n, 0))
    return im1_pts, im2_pts

# recovers homography via set of (p', p) pairs of corresponding points from two images
# im_pts are n-by-2 matrices holding (x, y) locations of n point correspondences
# H is (recovered 3x3 homography matrix)
def computeH(im1_pts, im2_pts):
    # A * h = b
    n = len(im1_pts)
    A = np.zeros((2*n, int(8)))
    b = np.zeros((2*n, int(1)))
    
    for i in range(n):
        pt1 = im1_pts[i]
        pt2 = im2_pts[i]
        # calculates B  
        b[i*2] = pt2[0]
        b[i*2+1] = pt2[1]

        # calc A
        A[i*2, 2] = 1
        A[i*2+1, 5] = 1

        A[i*2, 0] = pt1[0]
        A[i*2, 1] = pt1[1]
        A[i*2+1, 3] = pt1[0]
        A[i*2+1, 4] = pt1[1]

        A[i*2, 6] = -pt2[0] * pt1[0]
        A[i*2, 7] = -pt2[0] * pt1[1]
        A[i*2+1, 6] = -pt2[1] * pt1[0]
        A[i*2+1, 7] = -pt2[1] * pt1[1]

    H = np.linalg.lstsq(A,b, rcond=None)[0]
    H = np.vstack((H, [1]))
    H = H.reshape(3, 3)
    return H

# warps the im using the given homography matrix
def warpImage(im, H):
    imWarped = im
    # compute the BOUNDING BOX
    height, width, na = im.shape
    imCorners = np.array([[0,0], [height, width], [0, width], [height, 0]])
    
    max_x = -100000000
    max_y = -100000000
    min_x = 100000000
    min_y = 100000000

    # FOR EACH CORNER: find warped corners
    for corner in imCorners:
        y = corner[0]
        x = corner[1]
        ## warp four corners of original image, keep min and max values
        warped = H.dot(np.array([x, y, 1]))
        warped = warped/warped[2]
        warp_x = int(warped[0])
        warp_y = int(warped[1])

        if warp_y < min_y:
            min_y = warp_y
        if warp_y > max_y:
            max_y = warp_y
        if warp_x < min_x:
            min_x = warp_x
        if warp_x > max_x:
            max_x = warp_x
    ## size of box = max values (x, y) - min values (x, y)
    bound_y = max_y - min_y
    bound_x = max_x - min_x
    warpCorners = [min_x, max_x, min_y, max_y]
    # out of bounds indexes: mask to black 0 pixel
    imWarped = np.zeros((bound_y, bound_x, 3))
    for new_x in range(bound_x):
        for new_y in range(bound_y):
            # INVERSE WARP
            warped = np.linalg.inv(H).dot(np.array([new_x + min_x, new_y + min_y, 1]))
            warped = warped/warped[2]
            x = int(warped[0])
            y = int(warped[1])
            # if IN BOUNDS: translate pixel value from original to warped
            if (y >= 0 and y < height and x >= 0 and x < width):
                imWarped[new_y, new_x, :] = im[y, x, :]
    return imWarped, warpCorners

# merges image 1 and image 2 together using image 2 as the base into a mosaic
def mosaic(im1, im2, H):
    height, width, na = im1.shape
    imWarped, warpCorners = warpImage(im1, H)
    # retrieves min/max x corners from warped image
    min_x = warpCorners[0]
    max_x = warpCorners[1]
    min_y = warpCorners[2]
    max_y = warpCorners[3]
    imCorners = np.array([[0,0], [height, width], [0, width], [height, 0]])
    # for each corner: adjust boundary if needed
    for corner in imCorners:
        y = corner[0]
        x = corner[1]
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
    bound_y = max_y - min_y
    bound_x = max_x - min_x
    # positions images for merging
    mergeIm1 = np.zeros((bound_y, bound_x, 3))
    mergeIm2 = np.zeros((bound_y, bound_x, 3))
    # iterate over every pixel in new bound
    for new_x in range(min_x, max_x):
        for new_y in range (min_y, max_y):
            baseHeight, baseWidth, na = im2.shape
            warpMin_x = warpCorners[0]
            warpMax_x = warpCorners[1]
            warpMin_y = warpCorners[2]
            warpMax_y = warpCorners[3]
            # if in bounds of warp image
            if (new_y >= warpMin_y and new_y < warpMax_y and new_x >= warpMin_x and new_x < warpMax_x):
                mergeIm1[new_y - min_y, new_x - min_x, :] = imWarped[new_y - warpMin_y, new_x - warpMin_x, :]
                a, b, c = imWarped[new_y - warpMin_y, new_x - warpMin_x, :]
                if a == 0 or b == 0: # if black mask
                    # if in bounds of base image: set pixel value to base
                    if (new_y >= 0 and new_y < baseHeight and new_x >= 0 and new_x < baseWidth):
                        mergeIm2[new_y - min_y, new_x - min_x, :] = im2[new_y, new_x, :]
                elif (new_y >= 0 and new_y < baseHeight and new_x >= 0 and new_x < baseWidth):
                    # OVERLAPPING PORTION
                    mergeIm2[new_y - min_y, new_x - min_x, :] = im2[new_y, new_x, :]
            elif (new_y >= 0 and new_y < baseHeight and new_x >= 0 and new_x < baseWidth):
                # if NOT in bounds of warp but in bounds base image: set to base image
                mergeIm2[new_y - min_y, new_x - min_x, :] = im2[new_y, new_x, :]


    # LAPLACIAN BLENDS TOGETHER BOTH MERGES: 2 level
    mergeIm1 = sk.img_as_float(mergeIm1)
    mergeIm2 = sk.img_as_float(mergeIm2)

    gstack_im1 = gauss_stack(mergeIm1, 3)
    lstack_im1 = lap_stack(gstack_im1, 3)

    gstack_im2 = gauss_stack(mergeIm2, 3)
    lstack_im2 = lap_stack(gstack_im2, 3)
    im_m = skio.imread('assets/images/mask.jpg')/255
    im_m = sk.img_as_float(im_m)
    gstack_mask = gauss_stack(im_m, 3)

    im1_collapsed = np.zeros((bound_y, bound_x, 3))
    im2_collapsed = np.zeros((bound_y, bound_x, 3))
    # Creates, saves, and displays the lap stacks for the two images separate and combined
    for a, o, m in zip(lstack_im1, lstack_im2, gstack_mask):
        a = a * m
        o = o * (1-m)
        im1_collapsed = im1_collapsed + a;
        im2_collapsed = im2_collapsed + o;
        mergeIm = o + a

    # SAVES/DISPLAYS COMBINED ORAPPLE
    mergeIm = im2_collapsed + im1_collapsed
    skio.imshow(mergeIm)
    skio.show()
    return mergeIm

# PART B : sample code provided
def get_harris_corners(im, edge_discard=20):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 20

    # find harris corners
    h = corner_harris(im, method='eps', sigma=2)
    coords = peak_local_max(h, min_distance=1, indices=True)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    return h, coords


def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """
    
    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert(dimx == dimc, 'Data dimension does not match dimension of centers')

    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
            np.ones((   ndata, 1)) * np.sum((c**2).T, axis=0)    - \
            2 * np.inner(x, c)

# Adaptive Non-Maximal Suppression : returns list of top 500 points from harris
def ANMS(h, coords):
    # work in progress: chooses random 500 points
    # sort list of points
    coords = coords[:500, :500]
    # only take top 500 
    return coords

# takes a sample of 40 pixels around each point, then subsampels to 8x8 and returns
def feature(h, coords):
    all_patches = []
    for point in coords:
        x = point[1]
        y = point[0]

        patch = h[x - 20: x + 20, y - 20:y + 20]
        # reshape into 8x8
        patch = patch[0::5, 0::5]
        # normalize
        patch = (patch - np.min(patch)) / (np.max(patch) - np.min(patch))
        all_patches.append(patch)
    return all_patches

# Given a list of patches and coordinates for each image, it finds correct matches based on feature
def match_feature(patch1, patch2, coords1, coords2):
    dist = dist2(patch1, patch2)
    # UNFINISHED !!
    
# gets sets of n points from two images and saves
im1 = skio.imread('assets/images/mosaic1.JPG', as_gray=True)
im2 = skio.imread('assets/images/mosaic2.JPG', as_gray=True)
im1 = sk.img_as_float(im1)
im2 = sk.img_as_float(im2)

#pts1, pts2 = get_points(im1, im2, 10)
#np.save('blue1_pts', pts1)
#np.save('blue2_pts', pts2)

# LOAD POINTS 
#pts1 = np.load('blue1_pts.npy')
#pts2 = np.load('blue2_pts.npy')

im11, pts1 = get_harris_corners(im1)
pts11 = ANMS(im1, pts1)
im22, pts2 = get_harris_corners(im2)
pts22 = ANMS(im2, pts2)

# DISPLAY IMAGES AND POINTS#
plt.imshow(im1, cmap=plt.cm.gray)
plt.scatter(pts1[1, :], pts1[0, :], color="red")
plt.show()

plt.imshow(im1, cmap=plt.cm.gray)
plt.scatter(pts11[1, :], pts11[0, :], color="red")
plt.show()

plt.imshow(im2, cmap=plt.cm.gray)
plt.scatter(pts2[1, :], pts2[0, :], color="red")
plt.show()

plt.imshow(im2, cmap=plt.cm.gray)
plt.scatter(pts22[1, :], pts22[0, :], color="red")
plt.show()

# subsamples of patches
#feature(im1, pts11)

# PART A : RECTIFICATION
    # define pts2 by hand: for rectifications
    #pts2 = np.array([[145, 270], [592, 270], [592, 550], [145, 550]]) # trash pts
    #pts2 = np.array([[290, 175], [745, 175], [745, 600], [290, 600]]) # street pts
    #pts2 = np.array([[510, 175], [745, 175], [745, 602], [510, 602]]) # street pts
    # display pts2
    #plt.imshow(im2)
    #plt.scatter(pts2[:, 0], pts2[:, 1], color="red")
    #plt.show()

    # for image rectification: 
    #H = computeH(pts1, pts2)
    ##imWarp = warpImage(im1, im2, H)[0]
    #imMosaic = mosaic(im1, im2, H)

# PART A: MOSAICS
    #H = computeH(pts1, pts2)
    #imMosaic = mosaic(im1, im2, H)
    #plt.imshow(imMosaic)
    #plt.show()
    #saves warped image
    #plt.imsave('mosaic.jpg', imMosaic)
