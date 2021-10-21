# image_segmentation.py
"""Image Segmentation.
R. Connor Franckowiak
Nov 8, 2020
"""

import numpy as np
import scipy.linalg as la
from imageio import imread
from matplotlib import pyplot as plt
from scipy.sparse import lil_matrix as lm
from scipy.sparse import csc_matrix as csc
from scipy.sparse import linalg
from scipy.sparse import csgraph
from scipy.sparse import diags
import scipy.sparse


def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    #Returns A subtracted from the matrix formed by
    #the sum of the columns along the diagonal
    return np.diag(A.sum(axis=0))-A


def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    L = laplacian(A)
    #Find the Eigenvalues of the Laplacian of A
    eigvals = np.real(la.eigvals(L))
    count = 0
    smallest_e = eigvals[0]
    second_smallest = eigvals[0]
    for e in eigvals:
        print(e,smallest_e,second_smallest)
        if e < tol:
            #Count the total number of "zeros" (or negligible eigenvalues)
            count += 1
            e = 0
        #When a smallest eigenvalue is found, store it in smallest_e and
        #put the previous smallest value in second_smallest_e
        if e <= smallest_e:
            second_smallest = smallest_e
            smallest_e = e
        elif e < second_smallest:
            second_smallest = e
    return count, second_smallest


def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


class ImageSegmenter:
    """Class for storing and segmenting images.

    Attributes:
        image ((m,n,d) ndarray): original image matrix
        flat_grayscale ((,n) ndarray): flat array of the grayscale image.
    """


    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        image = imread(filename)
        self.image = image
        #Convert into grayscale, with values between 0 and 1
        if image.ndim == 3:
            grayscale = image.mean(axis=2) / 255
        else:
            grayscale = image / 255
        flat_grayscale = np.ravel(grayscale)
        self.flat_grayscale = flat_grayscale


    def show_original(self):
        """Display the original image."""
        #If in color, display normally
        if self.image.ndim == 3:
            plt.imshow(self.image)
        #Else if gray, then specify gray
        else:
            plt.imshow(self.image, cmap="gray")
        plt.axis("off")
        plt.show()



    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        #Extract size of image matrix
        if self.image.ndim == 3:
            m,n,d = self.image.shape
        else:
            m,n = self.image.shape
        #Construct the adjacency graph A
        A = lm((m*n,m*n),dtype=np.float)
        #Initialize the diagonal graph D
        D = np.array([0.]*(m*n))
        for i in range(m*n):
            neighbors, distances = get_neighbors(i,r,m,n)
            sum_weights = 0
            #Using equation 5.3 to construct the entries of A
            weights = np.exp(np.divide(-1*abs(self.flat_grayscale[neighbors]+(-self.flat_grayscale[i])),sigma_B2)-(np.divide(distances,sigma_X2)))
            A[i,neighbors] = weights
            D[i] = np.sum(weights)
        #Convert A to CSC Matrix for faster computation.
        A = A.tocsc()
        #Return the adjacency and diagonal matrices
        return A, D



    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        L = csgraph.laplacian(A)
        #Calculate D^(-1/2)
        D_sqrt = diags(np.reciprocal(np.sqrt(D).astype(float)))
        ev = linalg.eigsh(D_sqrt @ L @ D_sqrt,which="SM",k=2)[1][:,1]
        if self.image.ndim == 3:
            m,n,d = self.image.shape
        else:
            m,n = self.image.shape
        ev = np.reshape(ev,(m,n))
        mask = ev > 0
        return mask



    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        A, D = self.adjacency(r,sigma_B,sigma_X)
        mask = self.cut(A,D)
        #Show Original Image
        ax1 = plt.subplot(1,3,1)
        if self.image.ndim == 3:
            ax1.imshow(self.image)
        else:
            ax1.imshow(self.image, cmap="gray")
        #Stack mask if color image
        if self.image.ndim == 3:
            mask = np.dstack((mask, mask, mask))
        #Print one segment
        ax2 = plt.subplot(1,3,2)
        if self.image.ndim == 3:
            ax2.imshow(self.image*mask)
        else:
            ax2.imshow(self.image*mask, cmap="gray")
        #Print next segment
        ax3 = plt.subplot(1,3,3)
        if self.image.ndim == 3:
            ax3.imshow(self.image*~mask)
        else:
            ax3.imshow(self.image*~mask, cmap="gray")

        ax1.axis("off")
        ax2.axis("off")
        ax3.axis("off")
        plt.show()


if __name__ == '__main__':
    #ImageSegmenter("dream_gray.png").segment()
    ImageSegmenter("dream.png").segment()
    #ImageSegmenter("monument_gray.png").segment()
    #ImageSegmenter("blue_heart.png").segment()
