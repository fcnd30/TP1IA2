# Scikit-Image
from skimage.feature import graycomatrix, graycoprops
# Import Bitdesc
from BiT import bio_taxo
import mahotas.features as features
import cv2

def glcm(data):
    glcm = graycomatrix(data, [2], [0],None, symmetric=True,normed=True)
    diss = graycoprops(glcm, 'dissimilarity')[0,0]
    cont = graycoprops(glcm, 'contrast')[0,0]
    corr = graycoprops(glcm, 'correlation')[0,0]
    ener = graycoprops(glcm, 'energy')[0,0]
    homo = graycoprops(glcm, 'homogeneity')[0,0]
    return [diss, cont, corr, ener, homo]

def Bitdesc(data):
    features = bio_taxo(data)
    return features

def haralick (data): 
 	# data = cv2.imread(img)
 	return features.haralick(data).mean(0).tolist()

def bitdesc_glcm(image_path):
    return Bitdesc(image_path) + glcm(image_path)

def haralick_bitdesc(image_path):
    return haralick(image_path) + Bitdesc(image_path)










