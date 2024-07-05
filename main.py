import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from distance import retrieve_similar_images
from data_processing import extract_features, save_image, process_datasets

def main():
    st.write("Application de reconnaissance d'images")
    
    # Configuration du répertoire pour les images
    img_folder = "images"
    os.makedirs(img_folder, exist_ok=True)
    
    # Options de la barre latérale pour le type de descripteur
    descriptor_type = st.sidebar.radio(
        "Choisissez votre type de descripteur",
        ("Glcm", "haralick", "Bitdesc", "bitdesc_glcm", "haralick_bitdesc"),
        index=0
    )

    if st.sidebar.button("Process Datasets"):
        process_datasets('./datasets', descriptor_type)

    # Options de la barre latérale pour les métriques de distance
    distance_option = st.sidebar.radio(
        "Choisissez votre méthode de distance",
        ("euclidean", "manhattan", "chebyshev", "canberra"),
        index=0
    )

    # Option pour choisir le nombre de résultats à afficher
    num_results = st.sidebar.slider("Nombre d'images similaires à afficher", min_value=1, max_value=50, value=10)

    # Section pour télécharger et afficher l'image
    img_upload = st.file_uploader("Choisir une image", type=['png', 'jpeg', 'jpg', 'bmp'])
    if img_upload is not None:
        # Sauvegarder l'image téléchargée et extraire les caractéristiques
        img_path = save_image(img_upload, img_folder)
        features = extract_features(img_path, descriptor_type.lower())
        display_image = Image.open(img_path)

        # Chargement des signatures précalculées
        signatures = np.load(f'signatures_{descriptor_type}.npy')

        # Récupération des images similaires
        result = retrieve_similar_images(features_db=signatures, query_features=features, distance=distance_option, num_results=num_results)

        # Affichage des images similaires en grille
        columns = st.columns(4)  
        dataset_counts = {}
        index = 0
        for entry in result:
            if isinstance(entry, tuple) and len(entry) > 0:
                img_path = os.path.join('.\\datasets\\', entry[0])
                dataset_name = img_path.split('\\')[2]  # Ensure this matches your directory structure
                dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
                
                with open(img_path, "rb") as file:
                    image_data = file.read()
                columns[index % 4].image(image_data, caption=f'Image from {dataset_name}')
                index += 1
            else:
                st.error(f"Invalid format: {entry}")

        # Display a histogram if there are any dataset counts
        if dataset_counts:
            plt.figure(figsize=(10, 4))
            datasets, counts = zip(*dataset_counts.items())
            total_images = sum(counts)
            percentages = [(count / total_images) * 100 for count in counts]
            plt.bar(datasets, percentages, color='orange')
            plt.title('Distribution of Similar Images by Dataset')
            plt.xlabel('Dataset')
            plt.ylabel('Pourcentage (%)')
            st.pyplot(plt)


if __name__ == '__main__':
    main()








