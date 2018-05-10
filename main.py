"""
Author: Suraj Regmi
Date: 10th May, 2018
Description: Does image compression using singular value decomposition.
"""
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import numpy as np


def get_greyscale_image():
    """
    Get name of the file from the user and returns the
    image object after converting in greyscale mode.
    """
    filename = input("Enter filename of the image with extension:\n")
    image = Image.open(filename).convert('L')
    return image


def show_image(image):
    """Show the image in the screen of the entered file."""
    image.show()


def get_numpy_array(image):
    """Get numpy array for the image"""
    array = np.asarray(image)
    # n = original_image_array.shape[0]
    return array


def svd_decompose(array):
    """
    Returns singular value decomposition for the array
    """
    return np.linalg.svd(array)


def get_compressed_array(svd_factors, k):
    """
    Takes svd_factors of the matrix and no of key features and returns
    the compressed array
    :param svd_factors: Tuple having (U, S, V) of singular value decomposition
    :param k: no of most important features to be taken
    :return: compressed array from k features
    """
    u, s, v = svd_factors
    return np.matmul(u[:, :k].reshape(-1, k), np.matmul(np.diag(s[:k]), v[:k, :].reshape(k, -1)))


def write_text_and_show(image, n, k):
    """
    For writing text in the image giving total number of features(n)
    and used number of features(k).
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 28, encoding="unic")
    text = "Original number of features: {}\nUsed number of features: {}\nFeatures Compression Ratio: {}:{}"\
        .format(n, k, n, k)
    draw.text((10, 10), text, 255, font=font)

    # Show image in the screen
    image.show()


def main():
    original_image = get_greyscale_image()
    show_image(original_image)
    original_image_array = get_numpy_array(original_image)

    n = original_image_array.shape[0]  # n represents no of features in the image or no of rows of pixels
    print("\nThe original image is shown in the screen. \nYour original image has {} features.\n".format(n))
    k = int(input("Please enter the number of main features "
                  "you would like to put in the given image.\n"))

    compressed_image_array = get_compressed_array(svd_decompose(original_image_array), k)

    # Convert the compressed array to unsigned integer form in numpy
    compressed_image_array = compressed_image_array.astype('uint8')

    # Construct image object from the compressed array
    compressed_image = Image.fromarray(compressed_image_array)

    print("\nThe compressed image is displayed on the window.\n")

    write_text_and_show(compressed_image, n, k)


if __name__ == "__main__":
    main()
