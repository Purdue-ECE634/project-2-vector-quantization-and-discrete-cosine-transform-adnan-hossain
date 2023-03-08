import argparse
import os
import numpy as np
import cv2
import time
import pickle
import tempfile
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from zigzag import zigzag, inverse_zigzag


def compute_psnr(a, b):
    """
    a: image as a numpy array
    b: image as a numpy array

    return: PSNR between a and b

    """
    mse = np.mean((a - b)**2).item()
    return -10 * math.log10(mse) + 20 * math.log10(255-1)

def compute_msssim(a, b):
    """
    a: image as a numpy array
    b: image as a numpy array

    return: structural similarity index measure between a and b

    """
    return ssim(a, b, channel_axis=None)



def dctTransform(matrix):

    pi = 3.142857
    m = 8
    n = 8
 
    # dct will store the discrete cosine transform
    dct = []
    for i in range(m):
        dct.append([None for _ in range(n)])
 
    for i in range(m):
        for j in range(n):
 
            # ci and cj depends on frequency as well as
            # number of row and columns of specified matrix
            if (i == 0):
                ci = 1 / (m ** 0.5)
            else:
                ci = (2 / m) ** 0.5
            if (j == 0):
                cj = 1 / (n ** 0.5)
            else:
                cj = (2 / n) ** 0.5
 
            # sum will temporarily store the sum of
            # cosine signals
            sum = 0
            for k in range(m):
                for l in range(n):
 
                    dct1 = matrix[k][l] * math.cos((2 * k + 1) * i * pi / (
                        2 * m)) * math.cos((2 * l + 1) * j * pi / (2 * n))
                    sum = sum + dct1
 
            dct[i][j] = ci * cj * sum
    
    return dct


def get_DCT_img(img, N):
    """
    Function that extracts the required DCT coefficients
    """

    h, w = img.shape
    num_X_blocks = w//N
    num_Y_blocks = h//N
    DCT_img = np.zeros(img.shape)

    for i in range(num_Y_blocks):
        for j in range(num_X_blocks):       
            img_block = img[i*N:i*N+N, j*N:j*N+N]
            DCT_img[i*N:i*N+N, j*N:j*N+N] = dctTransform(img_block)      # cv2.dct(img_block, cv2.DCT_INVERSE)   <- {use this for faster code} 

    return DCT_img


def select_coefficients(DCT_img, N, K):
    """
    Function that selects DCT coefficients to be used for image approximation
    """ 

    filtered_DCT_img = np.zeros((DCT_img.shape))

    h, w = DCT_img.shape
    num_X_blocks = w//N
    num_Y_blocks = h//N
    img = np.zeros(DCT_img.shape)

    for i in range(num_Y_blocks):
        for j in range(num_X_blocks):
            img_block = DCT_img[i*N:i*N+N, j*N:j*N+N]
            DCT_list = zigzag(img_block)
            DCT_list[K:] = 0
            filtered_img_block = inverse_zigzag(DCT_list, N, N)
            filtered_DCT_img[i*N:i*N+N, j*N:j*N+N] = filtered_img_block

    return filtered_DCT_img


def img_from_DCT_img(DCT_img, N):
    """
    Function that creates image from DCT coefficients
    """

    h, w = DCT_img.shape
    num_X_blocks = w//N
    num_Y_blocks = h//N
    img = np.zeros(DCT_img.shape)

    for i in range(num_Y_blocks):
        for j in range(num_X_blocks):       
            img_block = DCT_img[i*N:i*N+N, j*N:j*N+N]
            img[i*N:i*N+N, j*N:j*N+N] = cv2.idct(img_block)

    return img


def main():
    parser = argparse.ArgumentParser(description="Image Approximation using Discrete Cosine Transform (DCT) coefficients")
    parser.add_argument("--image_path", type=str, default="sample_image/watch.png", help="Path to the image on which DCT is performed")
    parser.add_argument("--block_size", type=int, default=8, help="Size of the blocks the image is divided into")
    parser.add_argument("--DCT_coefficients", type=int, default=2, help="Size of each codeword in each dimension")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    _, img_name = args.image_path.split('/')
    img_name, _ = img_name.split('.')                                                                       # extracting the name of the image

    img = cv2.imread(args.image_path, 0).astype(float)                                                      # importing the image
    N = args.block_size                                                                                     # size of DCT matrix
    K = args.DCT_coefficients                                                                               # Number of DCT coefficients to use

    DCT_img = get_DCT_img(img, N)                                                                           # Creating the DCT image

    filtered_DCT_img = select_coefficients(DCT_img, N, K)                                                   # DCT image with only K coefficients
    rec_img = img_from_DCT_img(filtered_DCT_img, N)                                                         # reconstructed image with only K coefficients

    # Displaying and saving the original and reconstructed images
    os.makedirs("images_DCT", exist_ok=True)
    cv2.imshow('Original Image', img.astype(np.uint8))
    cv2.imshow(f'Approximate Image K={K}', rec_img.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f"images_DCT/Original_Image_{img_name}.jpg", img)
    cv2.imwrite(f"images_DCT/Approximate_Image_{img_name}_K{K}.jpg", rec_img)

    # Calculating PSNR and SSIM
    PSNR = compute_psnr(img, rec_img)
    print(f"PSNR of approximate image with respect to original image: {PSNR}")
    SSIM = compute_msssim(img, rec_img)
    print(f"MS-SSIM of approximate image relative to its original image: {SSIM}")

    # Saving the experiment results
    os.makedirs(f'results_DCT', exist_ok=True)
    with open(f'results_DCT/{img_name}_K{K}.txt', 'w') as f:
        f.write(f"PSNR of aaproximate image with respect to original image: {PSNR}\n")
        f.write(f"MS-SSIM of approximate image relative to its original image: {SSIM}\n")
    

if __name__ == '__main__':
    main()