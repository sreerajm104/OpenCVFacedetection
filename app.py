# Making the necessary imports
import streamlit as st
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
# imginput = cv2.imread("ministers.jpg")

st.title("File Upload and OpenCV Uses")
st.markdown("Trying the Face Detection from uploaded image")

st.sidebar.title("File Upload Testing")
st.sidebar.markdown("Using OpenCV Face Classifier the faces are detected in the user uploaded image")

uploaded_file=st.sidebar.file_uploader(label="Upload Image",type=["jpg","jpeg","png"],key="1")
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
    
    image = cv2.imdecode(file_bytes, 1)
    grayimg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_classifier.detectMultiScale(grayimg,1.3,5)
    if faces != ():
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),color=(0, 255, 0), thickness=3)
        st.success("Faces Detected")
    
        st.image(image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR),width=600,use_column_width=800)
    else:
        st.success("No Faces Identified")
        
    