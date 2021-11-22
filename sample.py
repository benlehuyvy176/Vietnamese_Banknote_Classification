import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
from PIL import Image, ImageOps
menu = ['Read Data', 'Show Webcam']

choice = st.sidebar.selectbox('Money Classification', menu)

if choice=='Read Data':
    st.title('Please upload your money picture!')
    st.image('media\Vietnames-Dong-500-000-to-10-000.jpg')

    class_names = ['1000','10000','100000','2000','20000','200000','5000','50000','500000']
    @st.cache(allow_output_mutation=True)
    def load_model():
        model=tf.keras.models.load_model('Model.h5')
        return model
    with st.spinner('Model is being loaded..'):
        model=load_model()

    st.write("""
            # Money Classification
            """
            )

    filename = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    def import_and_predict(image_data, model):
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
    if filename is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(filename)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, model)
        score = tf.nn.softmax(predictions[0])
        st.write(predictions)
        st.write(score)
        print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

elif choice=='Show Webcam':
    st.title('Open your webcam')
    st.warning('Webcam show on local computer ONLY')
    class_names = ['1000','10000','100000','2000','20000','200000','5000','50000','500000']
    @st.cache(allow_output_mutation=True)
    def load_model():
        model=tf.keras.models.load_model('Model.h5')
        return model
    with st.spinner('Model is being loaded..'):
        model=load_model()

    st.write("""
            # Money Classification
            """
            )
    show = st.checkbox('Show!')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0) # device 1/2

    while show:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    else:
        camera.release()
    
    while True:
        try:
            check, frame = webcam.read()
            print(check) #prints true as long as the webcam is running
            print(frame) #prints matrix values of each framecd 
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'): 
                cv2.imwrite(filename='saved_img.jpg', img=frame)
                webcam.release()
                img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
                img_new = cv2.imshow("Captured Image", img_new)
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
                print("Processing image...")
                img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
                print("Converting RGB image to grayscale...")
                gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
                print("Converted RGB image to grayscale...")
                # print("Resizing image to 28x28 scale...")
                # img_ = cv2.resize(gray,(28,28))
                # print("Resized...")
                img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
                print("Image saved!")
            
                break
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
        
        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

        if filename is None:
            st.text("Please take a picture")
        else:
            image = Image.open(filename)
            st.image(image, use_column_width=True)
            predictions = import_and_predict(image, model)
            score = tf.nn.softmax(predictions[0])
            st.write(predictions)
            st.write(score)
            print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )