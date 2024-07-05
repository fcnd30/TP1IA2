import os
import cv2
import cv2
print(cv2.__version__)

import numpy as np
import streamlit as st

from descripteurs import glcm, Bitdesc, haralick, bitdesc_glcm, haralick_bitdesc

def extract_features(image_path, descriptor_type):
    """
    Extract features from a grayscale image based on the specified descriptor.
    Args:
        image_path (str): Path to the image file.
        descriptor_type (str): Type of descriptor to use ('glcm', 'Haralick', 'bitdesc', 'bitdesc_glcm', 'haralick_bitdesc').
    Returns:
        np.array: Extracted features.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        if descriptor_type == 'glcm':
            return glcm(img)
        elif descriptor_type == 'haralick':
            return haralick(img)
        elif descriptor_type == 'bitdesc':
            return Bitdesc(img)
        elif descriptor_type == 'bitdesc_glcm':
            return bitdesc_glcm(img)
        elif descriptor_type == 'haralick_bitdesc':
            return haralick_bitdesc(img)
    else:
        raise FileNotFoundError("Image file not found or could not be opened.")

def process_datasets(root_folder, descriptor_type):
    """
    Process each dataset folder within the root folder and extract features using the specified descriptor,
    if their signature file does not already exist.
    Args:
        root_folder (str): Root folder containing all the datasets.
        descriptor_type (str): Descriptor to use for feature extraction.
    """
    signature_path = f'signatures_{descriptor_type}.npy'
    if os.path.exists(signature_path):
        print(f'Signature file {signature_path} already exists. Skipping processing for {descriptor_type}.')
        return

    all_features = []  # List to store all features and metadata
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Construct relative path and extract features
                relative_path = os.path.relpath(os.path.join(root, file), root_folder)
                file_name = f'{relative_path.split("/")[0]}_{file}'
                features = extract_features(os.path.join(root, file), descriptor_type)
                # Extract class name from the relative path
                class_name = os.path.basename(os.path.dirname(relative_path))
                # Append features, class name, and relative path to the list
                print(f'File: {file_name} -> Path: {relative_path} -> Features : {features}')
                all_features.append(np.concatenate((features, [class_name, relative_path])))
    signatures = np.array(all_features)
    np.save(signature_path, signatures)
    print(f'Features successfully stored for {descriptor_type} in {signature_path}')

def save_image(img_upload, img_folder):
    if img_upload is not None:
        # Create a file path for the uploaded file in the specified folder
        img_path = os.path.join(img_folder, img_upload.name)
        # Open the file in binary write mode and write the image data
        with open(img_path, "wb") as f:
            f.write(img_upload.getbuffer())
        return img_path
    else:
        return None   

def main():
    descriptors = ['glcm', 'Haralick', 'bitdesc', 'bitdesc_glcm', 'haralick_bitdesc']
    for descriptor_type in descriptors:
        process_datasets('./datasets', descriptor_type)

if __name__ == '__main__':
    main()

                
                




                
                



