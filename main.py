import cv2
import numpy as np  
import streamlit as st
from src.services.PreprocessImage import preprocess_image
from src.model.Image import Image
from src.services.computeDistance import compute_distance


def main():
    img_path="./test/img_test/img1.jpeg"
    img_input=cv2.imread(img_path)
    
    img_object=Image(img_input=img_input)
    img_object.preprocess()
    
    feature_img_input=img_object.ExtractFeatures()

    img_path2="./test/img_test/img5.jpeg"
    img_input2=cv2.imread(img_path2)
    img_object2=Image(img_input=img_input2)
    img_object2.preprocess()
    feature_img_input2=img_object2.ExtractFeatures()

    distance=compute_distance(img_object, img_object2)
    print(distance)

if __name__ == "__main__":
    main()