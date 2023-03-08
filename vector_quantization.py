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

def get_object_size(obj, unit='bits'):
    assert unit == 'bits'
    with tempfile.TemporaryFile() as fp:
        pickle.dump(obj, fp)
        num_bits = os.fstat(fp.fileno()).st_size * 8
    return num_bits


def create_dataset(train_path, test_img_path):
    """
    Creates the dataset for training and testing

    train_path: path to the directory containg trainign images
    test_img_path: path to the test image
    return: a tuple of numpy arrays containing training images and test image

    """

    train_list = os.listdir(train_path)
    train_imgs = []
    for item in train_list:
        img = cv2.imread(os.path.join(train_path, item), 0)
        img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_AREA)
        train_imgs.append(img)
    
    train_imgs = np.array(train_imgs).astype(float)
    test_img = cv2.imread(test_img_path, 0).astype(float)
    test_img = cv2.resize(test_img, (512, 512), interpolation = cv2.INTER_AREA)

    return train_imgs, test_img


def find_codeword_idx(img_block, code_book):
    """
    Function used to update which pixel block is assigned to which codeword
    img_block: numpy array containing a single image block
    code_book: numpy array containing the code book
    return: index of the codebook which contains the codeword that matches the image block
    """

    min_MSE = 1e7
    for i in range(code_book.shape[0]):
        codeword = code_book[i,:,:]
        MSE = np.square(np.subtract(img_block, codeword)).mean()
        if MSE < min_MSE:
            l = i
            min_MSE = MSE

    return l


def assign_codeword(imgs, code_book, N):
    """
    Function used for assigning a codeword to each sample of pixels
    """

    num_imgs, h, w = imgs.shape
    num_X_blocks = w//N
    num_Y_blocks = h//N

    code_map = np.zeros((num_imgs, num_Y_blocks, num_X_blocks), dtype=int)     

    for img in range(num_imgs):
        for i in tqdm(range(num_Y_blocks), desc=f"Assigning codeword to blocks of image {img}"):
            for j in range(num_X_blocks):       
                img_block = imgs[img, i*N:i*N+N, j*N:j*N+N]
                l = find_codeword_idx(img_block, code_book)
                code_map[img, i, j] = l

    return code_map


def reevaluate_codeword(imgs, code_map, L, N):
    """
    Recalculates the codebook depending on image blocks assigned to each codeword
    imgs: numpy array containing the batch of images
    code_map: numpy array  which contains the mapping between each image block and its associated codeword
    L: size of the codebook
    N: block size
    return: updated codebook
    """

    num_imgs, h, w = imgs.shape
    num_X_blocks = w//N
    num_Y_blocks = h//N
    code_book = np.zeros((L, N, N), dtype=float) 
    list_of_blocks = []

    for l in tqdm(range(L), desc=f"Re-evaluating codewords"):
        list_of_blocks = []
        for img in range(num_imgs):
            for i in range(num_Y_blocks):
                for j in range(num_X_blocks):     
                    if code_map[img, i, j] == l:
                        list_of_blocks.append(imgs[img, i*N:i*N+N, j*N:j*N+N])  
        
        if len(list_of_blocks) != 0:
            list_of_blocks = np.array(list_of_blocks)
            code_book[l,:,:] = np.rint(np.mean(list_of_blocks, axis=0))

    return code_book


def distortion(imgs, code_book, code_map, N):
    """
    Calcualtes the total MSE(distortion) between the image blocks and their assigned codewords
    imgs: numpy array containing the batch of images                                                           shape: (num_images, h, w)
    code_book: numpy array that contains all the codewords                                                     shape: (L, N, N)
    code_map: numpy array  which contains the mapping between each image block and its associated codeword     shape: (num_images, h//N, w//N)
    N: block size
    return: mean distortion between all the image blocks and their associated codewords
    """

    num_imgs, h, w = imgs.shape
    num_X_blocks = w//N
    num_Y_blocks = h//N
    D = 0

    for img in range(num_imgs):
        for i in tqdm(range(num_Y_blocks), desc=f"Calculating total distortion for image {img}"):
            for j in range(num_X_blocks):    
                img_block = imgs[img, i*N:i*N+N, j*N:j*N+N]
                l = code_map[img, i, j]
                codeword = code_book[l,:,:]
                MSE = np.square(np.subtract(img_block, codeword)).mean()
                D += MSE

    return D/(num_imgs*num_X_blocks*num_Y_blocks)


def train(train_imgs, L=128, N=4, num_epochs=1, img_name='0'):
    """
    Function used for creating the codebook
    train_imgs: numpy array consisting of the training images      shape: (num_images, w, h)
    L: size of the codebook
    N: size of each codeword in each dimension
    return: the codebook
    """

    l_max, l_min = train_imgs.max(), train_imgs.min()
    code_book = np.random.randint(low=l_min, high=l_max, size=(L, N, N)).astype(float)            # randomly initialize the codebook
    D_prev = 1e7
    D_list = []
    time_per_epoch = 0
    final_epoch = num_epochs
    start_time = time.time()
    for epoch in range(1, num_epochs+1):
        print(f"Epoch: {epoch}")
        start = time.time()
        code_map = assign_codeword(train_imgs, code_book, N)
        code_book = reevaluate_codeword(train_imgs, code_map, L, N)
        D = distortion(train_imgs, code_book, code_map, N)
        print(f"Distortion for epoch {epoch} = {D}")
        D_list.append(D)
        if abs(D-D_prev) < 0.01:
            print("Training has converged, so breaking from training loop")
            final_epoch = epoch
            break
        D_prev = D
        end = time.time()
        print(f"Epoch training time = {end-start} seconds")
        time_per_epoch += end-start
    end_time = time.time()
    
    time_per_epoch = time_per_epoch/final_epoch
    print(f"Training finished. Total training time = {(end_time-start_time)/60} minutes")

    plt.plot(range(1, final_epoch+1), D_list)
    plt.xlabel("Epoch")
    plt.ylabel("Distortion (Total MSE of all training images)")
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/training_curve_{img_name}_L{L}_imgs{train_imgs.shape[0]}_epochs{num_epochs}.jpg")
    plt.show()

    return code_book, time_per_epoch


def quantize(img, code_book, N):
    """
    This function quantizes an image for a given codebook
    """

    h, w = img.shape
    num_X_blocks = w//N
    num_Y_blocks = h//N
    quantized_img = np.zeros((h, w))

    code_map = assign_codeword(img.reshape(1, h, w), code_book, N).reshape(num_Y_blocks, num_X_blocks)
    for i in range(num_Y_blocks):
        for j in range(num_X_blocks):    
            l = code_map[i, j]
            quantized_img[i*N:i*N+N, j*N:j*N+N] = code_book[l,:,:]

    return quantized_img


def main():
    parser = argparse.ArgumentParser(description="Vector Quantization using Generalized Lloyd Algorithm")
    parser.add_argument("--train_dir", type=str, default="train", help="Path to the directory containing the images used for creating the codebook")
    parser.add_argument("--test_image_path", type=str, default="test/monarch.png", help="Path to the test image")
    parser.add_argument("--code_book_size", type=int, default=256, help="Size of the codebook used for quantization (Quantization Levels)")
    parser.add_argument("--code_word_size", type=int, default=4, help="Size of each codeword in each dimension")
    parser.add_argument("--num_epochs", type=int, default=50, help="How many epochs to train the algorithm for in order to calculate the code book")
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'eval'], help="Whether to do training or just evaluation")
    parser.add_argument("--checkpoint", type=str, default="codebook_L256_imgs10_epochs50.npy", help="name of the file containing the codebook")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    _, img_name = args.test_image_path.split('/')
    img_name, _ = img_name.split('.')                                                                                                                               # extracting the name of the image
    train_imgs, test_img = create_dataset(args.train_dir, args.test_image_path)                                                                                     # importing the train and eval images

    if args.mode == 'train':
        codebook, time_per_epoch = train(train_imgs, args.code_book_size, args.code_word_size, args.num_epochs, img_name)                                           # creating the codebook using train images
        os.makedirs("codebooks", exist_ok=True)
        np.save(os.path.join('codebooks', f'codebook_L{args.code_book_size}_imgs{train_imgs.shape[0]}_epochs{args.num_epochs}.npy'), codebook)                      # saving the codebook
    else:
        file_name = args.checkpoint 
        codebook = np.load(f'codebooks/{file_name}')                                                                                                                # loading already saved codebook

    quantized_img = quantize(test_img, codebook, args.code_word_size)                                                                                               # quantizing the test image using the codebook

    # Displaying and saving the original and quantized images
    os.makedirs("images", exist_ok=True)
    cv2.imshow('Original Image', test_img.astype(np.uint8))
    cv2.imshow('Quantized Image', quantized_img.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f"images/Quantized_image_{img_name}_L{args.code_book_size}_imgs{train_imgs.shape[0]}_epochs{args.num_epochs}.jpg", quantized_img)
    cv2.imwrite(f"images/Original_image_{img_name}_L{args.code_book_size}_imgs{train_imgs.shape[0]}_epochs{args.num_epochs}.jpg", test_img)

    # Calculating PSNR and SSIM
    PSNR = compute_psnr(quantized_img, test_img)
    print(f"PSNR of quantized image with respect to original image: {PSNR}")
    SSIM = compute_msssim(quantized_img, test_img)
    print(f"SSIM of the quantized image relative to its original image: {SSIM}")

    # Saving the evaluation results
    os.makedirs(f'results', exist_ok=True)
    with open(f'results/{img_name}_L{args.code_book_size}_imgs{train_imgs.shape[0]}_epochs{args.num_epochs}.txt', 'w') as f:
        f.write(f"Num training images = {train_imgs.shape[0]}\n")
        f.write(f"Codebook size = {args.code_book_size}\n")
        f.write(f"Num epochs trained = {args.num_epochs}\n")
        f.write(f"PSNR of quantized image with respect to original image: {PSNR}\n")
        f.write(f"SSIM of the quantized image relative to its original image: {SSIM}\n")
        if args.mode == 'train':
            f.write(f"Training time per epoch: {time_per_epoch} seconds")
    

if __name__ == '__main__':
    main()