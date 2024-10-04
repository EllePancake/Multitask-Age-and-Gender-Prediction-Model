import os
import cv2
import itertools
from PIL import Image
import imagehash
import matplotlib.pyplot as plt
import seaborn as sns
import random
from skimage import io, color
from skimage.feature import hog
import numpy as np
from sklearn.metrics import pairwise_distances


def display_top_images(images, sharpness_or_saturation, folder_path, title):
    """
    Displays a grid of the top images based on sharpness or saturation value.

    Args:
        images (DataFrame): A DataFrame containing image information.
        sharpness_or_saturation (str): The column name indicating the value
        used for ranking images.
        folder_path (str): The folder path containing the images.
        title (str): The title for the plot.

    Returns:
        None
    """
    plt.figure(figsize=(10, 10))
    for i, row in enumerate(images.iterrows(), 1):
        file_name = row[1]['file_name']
        sharp_or_sat_value = row[1][sharpness_or_saturation]
        img_path = os.path.join(folder_path, file_name)
        img = Image.open(img_path)
        plt.subplot(3, 3, i)
        plt.imshow(img)
        plt.title(f"{sharpness_or_saturation}: {sharp_or_sat_value:.2f}")
        plt.axis('off')
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def compute_hamming_distances(folder_path):
    """
    Computes Hamming distances between perceptual hashes of images in a folder.

    Args:
        folder_path (str): The folder path containing the images.

    Returns:
        list: A list of Hamming distances between image pairs.
    """
    image_hashes = {}
    distances = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".jpg"):
            file_path = os.path.join(folder_path, file_name)
            image = Image.open(file_path)
            hash_value = imagehash.phash(image)
            image_hashes[file_name] = hash_value

    for (file1, hash1), (file2, hash2) in itertools.combinations(
        image_hashes.items(), 2
    ):
        hamming_distance = hash1 - hash2
        distances.append(hamming_distance)

    return distances


def plot_smallest_hamming_distances(folder_path):
    """
    Plots the distribution of the smallest Hamming distances between image
    pairs.

    Args:
        folder_path (str): The folder path containing the images.

    Returns:
        None
    """
    distances = compute_hamming_distances(folder_path)

    min_distance = min(distances)

    plt.figure(figsize=(8, 6))
    sns.countplot(x=[min_distance], palette="Blues_r")
    plt.title('Counts of Smallest Hamming Distances Between Image Pairs')
    plt.xlabel('Smallest Hamming Distance')
    plt.ylabel('Count')
    plt.show()


def plot_hamming_distance_distribution(folder_path):
    """
    Plots a histogram of the Hamming distance distribution between image pairs.

    Args:
        folder_path (str): The folder path containing the images.

    Returns:
        None
    """
    distances = compute_hamming_distances(folder_path)

    plt.figure(figsize=(8, 6))
    sns.histplot(distances, bins=30, kde=False, color='steelblue')
    plt.title('Distribution of Hamming Distances Between Image Pairs')
    plt.xlabel('Hamming Distance')
    plt.ylabel('Frequency')
    plt.show()


def find_identical_images(folder_path):
    """
    Finds pairs of images in a given folder that are identical based on their
    perceptual hash.

    Args:
        folder_path (str): The folder path containing the images.

    Returns:
        list: A list of tuples, where each tuple contains the file names of
        two identical images.
    """
    image_hashes = {}
    identical_images = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".jpg"):
            file_path = os.path.join(folder_path, file_name)
            image = Image.open(file_path)
            hash_value = imagehash.phash(image)
            image_hashes[file_name] = hash_value

    for (file1, hash1), (file2, hash2) in itertools.combinations(
        image_hashes.items(), 2
    ):
        hamming_distance = hash1 - hash2
        if hamming_distance == 0:
            identical_images.append((file1, file2))

    return identical_images


def display_identical_images(identical_images, folder_path):
    """
    Displays groups of images that are identified as identical based on their
    perceptual hash.

    Args:
        identical_images (list): A list of tuples, where each tuple contains
        file names of identical images.
        folder_path (str): The folder path containing the images.

    Returns:
        None
    """
    for group in identical_images:
        plt.figure(figsize=(10, 5))
        for idx, file_name in enumerate(group):
            img_path = os.path.join(folder_path, file_name)
            image = Image.open(img_path)
            plt.subplot(1, len(group), idx + 1)
            plt.imshow(image)
            plt.title(file_name)
            plt.axis('off')
        plt.tight_layout()
        plt.show()


def hamming_similarity_percentage(hash1, hash2):
    """
    Calculates the similarity percentage between two image hashes based on
    Hamming distance.

    Args:
        hash1 (ImageHash): Perceptual hash of the first image.
        hash2 (ImageHash): Perceptual hash of the second image.

    Returns:
        float: Similarity percentage between the two images, from 0 - 100.
    """
    return 100 - ((hash1 - hash2) / len(hash1.hash) ** 2) * 100


def find_similar_images_by_thresholds(folder_path, similarity_ranges):
    """
    Finds groups of similar images based on defined similarity threshold
    ranges.

    Args:
        folder_path (str): The folder path containing the images.
        similarity_ranges (dict): A dictionary with keys as labels for the
        similarity ranges,
                                  and values as tuples representing the
                                  minimum and maximum similarity percentages.

    Returns:
        dict: A dictionary where each key is a similarity range label and the
        value is a list of tuples,
              each containing two file names and their similarity percentage.
    """
    image_hashes = {}
    similar_images = {
        range_label: [] for range_label in similarity_ranges.keys()
        }

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".jpg"):
            file_path = os.path.join(folder_path, file_name)
            image = Image.open(file_path)
            hash_value = imagehash.phash(image)
            image_hashes[file_name] = hash_value

    for (file1, hash1), (file2, hash2) in itertools.combinations(
        image_hashes.items(), 2
    ):
        similarity_percentage = hamming_similarity_percentage(hash1, hash2)

        for range_label, (min_similarity,
                          max_similarity) in similarity_ranges.items():
            if min_similarity <= similarity_percentage <= max_similarity:
                similar_images[range_label].append((file1,
                                                    file2,
                                                    similarity_percentage))
                break

    return similar_images


def display_similar_images(similar_image_groups, folder_path, title):
    """
    Displays pairs of similar images side by side in a 1x2 layout, with
    similarity percentage.

    Args:
        similar_image_groups (list): A list of tuples where each tuple
        contains two file names of similar images
                                     and their similarity percentage.
        folder_path (str): The folder path containing the images.
        title (str): The title for the entire plot.

    Returns:
        None
    """
    for group in similar_image_groups:
        file1, file2, similarity_percentage = group
        plt.figure(figsize=(10, 5))
        for idx, file_name in enumerate([file1, file2]):
            img_path = os.path.join(folder_path, file_name)
            image = Image.open(img_path)
            plt.subplot(1, 2, idx + 1)
            plt.imshow(image)
            plt.title(f"{file_name}\nSimilarity: {similarity_percentage:.2f}%")
            plt.axis('off')
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


def detect_faces(image_path):
    """
    Detects faces in an image using a pre-trained Haar Cascade classifier.

    Args:
        image_path (str): Path to the image file.

    Returns:
        bool: True if at least one face is detected in the image, else False.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                         'haarcascade_frontalface_default.xml')

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30))

    return len(faces) > 0


def flag_non_face_images(folder_path):
    """
    Checks all images in a given folder for faces and flags images that do not
    contain any faces.

    Args:
        folder_path (str): The folder path containing the images to be checked.

    Returns:
        list: A list of file names for images that do not contain any detected
        faces.
    """
    non_face_images = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            file_path = os.path.join(folder_path, file_name)
            if not detect_faces(file_path):
                non_face_images.append(file_name)

    return non_face_images


def display_outliers(outlier_file_names, folder_path, n_images=9):
    """
    Displays a grid of images that are considered outliers.

    Args:
        outlier_file_names (list): A list of file names of images that are
        considered outliers.
        folder_path (str): The folder path containing the images.
        n_images (int, optional): The number of outlier images to display.
        Default is 9.

    Returns:
        None
    """
    plt.figure(figsize=(10, 10))
    for i, file_name in enumerate(outlier_file_names[:n_images]):
        img_path = os.path.join(folder_path, file_name)
        img = Image.open(img_path)
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(file_name)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def display_images(image_list, folder_path, title):
    """
    Displays a grid of images from the specified folder.

    Args:
        image_list (list): A list of image file names to be displayed.
        folder_path (str): The folder path containing the images.
        title (str): The title for the entire plot.

    Returns:
        None
    """
    total_images = len(image_list)
    n_cols = 5
    n_rows = (total_images + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 3 * n_rows))
    for idx, image_name in enumerate(image_list):
        img_path = os.path.join(folder_path, image_name)
        img = Image.open(img_path)

        plt.subplot(n_rows, n_cols, idx + 1)
        plt.imshow(img)
        plt.title(image_name, fontsize=8)
        plt.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def compute_hog_features(image_path):
    """
    Computes Histogram of Oriented Gradients (HOG) features for an image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        ndarray: The HOG features of the image as a NumPy array.

    Functionality:
        - Loads the image from the specified path.
        - Converts the image to grayscale.
        - Computes HOG features using the specified block normalization and
        pixel size.
    """
    img = io.imread(image_path)
    gray_img = color.rgb2gray(img)
    hog_features = hog(gray_img, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    return hog_features


def detect_hog_outliers(folder_path, threshold=0.05, batch_size=100):
    """
    Detects outlier images in a folder based on HOG feature distances using
    batch processing.

    Args:
        folder_path (str): The folder path containing the images to be
        analyzed.
        threshold (float, optional): The distance threshold above which an
        image is considered an outlier. Default is 0.05.
        batch_size (int, optional): The number of images to process in each
        batch. Default is 100.

    Returns:
        list: A list of file names of images that are detected as outliers.

    Functionality:
        - Processes images in batches to compute HOG features.
        - Calculates pairwise distances between HOG features of images.
        - Detects images with a mean distance greater than the specified
        threshold as outliers.
        - Returns a list of file names of detected outlier images.
    """
    file_list = [
        file_name for file_name in os.listdir(folder_path)
        if file_name.endswith('.jpg') or file_name.endswith('.png')
    ]

    outliers = []
    n = len(file_list)

    for i in range(0, n, batch_size):
        batch_files = file_list[i:i+batch_size]
        hog_features_list = []

        for file_name in batch_files:
            file_path = os.path.join(folder_path, file_name)
            hog_feat = compute_hog_features(file_path)
            hog_features_list.append(hog_feat)

        hog_features_list = np.array(hog_features_list)

        distances = pairwise_distances(hog_features_list)

        for j, distance in enumerate(distances.mean(axis=1)):
            if distance > threshold:
                outliers.append(batch_files[j])

    return outliers


def show_image_pair(image1_path, image2_path):
    """
    Displays two images side by side.

    Args:
        image1_path (str): Path to the first image file.
        image2_path (str): Path to the second image file.

    Returns:
        None
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    image1 = Image.open(image1_path)
    axs[0].imshow(image1)
    axs[0].set_title("Image 1")
    axs[0].axis('off')

    image2 = Image.open(image2_path)
    axs[1].imshow(image2)
    axs[1].set_title("Image 2")
    axs[1].axis('off')

    plt.show()


def find_exact_duplicates(folder_path):
    """
    Finds pairs of images in a folder that are exactly identical based on
    their perceptual hash.

    Args:
        folder_path (str): The folder path containing the images to be checked.

    Returns:
        list: A list of tuples, where each tuple contains the file names of
        two identical images.

    Additional Functionality:
        - Prints the total number of duplicate pairs found.
        - Randomly selects and displays up to 3 pairs of duplicate images side
        by side.
    """
    image_hashes = {}
    duplicates = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".jpg"):
            file_path = os.path.join(folder_path, file_name)
            image = Image.open(file_path)

            hash_value = imagehash.phash(image)

            if hash_value in image_hashes:
                existing_file = image_hashes[hash_value]
                duplicates.append((existing_file, file_name))
            else:
                image_hashes[hash_value] = file_name

    print(f"Total number of 100% similar "
          f"(duplicate) pairs found:{len(duplicates)}")

    if len(duplicates) > 0:
        random_pairs = random.sample(duplicates, min(3, len(duplicates)))
        for pair in random_pairs:
            print(f"Displaying pair: {pair[0]} and {pair[1]}")
            show_image_pair(os.path.join(
                folder_path, pair[0]), os.path.join(folder_path, pair[1]))

    return duplicates


def delete_duplicates(duplicates, folder_path):
    """
    Deletes duplicate images from a folder, keeping only one copy of each
    duplicate pair.

    Args:
        duplicates (list): A list of tuples, where each tuple contains the
        file names of two identical images.
        folder_path (str): The folder path containing the images.

    Returns:
        None

    Functionality:
        - Iterates through each pair of duplicates.
        - Deletes the second file in each duplicate pair, if it exists.
        - Prints messages indicating whether a file was deleted or if it does
        not exist.
    """
    for pair in duplicates:
        file_to_delete = os.path.join(folder_path, pair[1])
        if os.path.exists(file_to_delete):
            print(f"Deleting {file_to_delete}")
            os.remove(file_to_delete)
        else:
            print(f"File {file_to_delete} does not exist.")


def display_outliers_in_batches(image_names, folder_path, batch_size=25):
    """
    Displays outlier images in batches, with a specified batch size.

    Args:
        image_names (list): A list of file names of images that are considered
        outliers.
        folder_path (str): The folder path containing the images.
        batch_size (int, optional): The number of images to display per batch.
        Default is 25.

    Returns:
        None

    Functionality:
        - Displays images in batches, with a grid layout (5x5) to show up to
        25 images per batch.
        - Iterates through all outlier images, displaying them in multiple
        figures if necessary.
        - If the number of images in a batch is less than the grid size, the
        extra axes are turned off.
    """
    total_outliers = len(image_names)
    num_batches = (total_outliers + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, total_outliers)
        current_batch = image_names[batch_start:batch_end]

        fig, axes = plt.subplots(5, 5, figsize=(15, 15))
        axes = axes.flatten()

        for ax, image_name in zip(axes, current_batch):
            img_path = os.path.join(folder_path, image_name)
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(image_name, fontsize=8)
            ax.axis('off')

        for remaining_ax in axes[len(current_batch):]:
            remaining_ax.axis('off')

        plt.tight_layout()
        plt.show()
