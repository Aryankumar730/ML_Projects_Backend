import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

new_model = tf.keras.models.load_model("Final_model_001.h5")

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

classes = ["angry","disgust","fear","happy","neutral","sad","surprise"]

result_dict = {}
for i, item in enumerate(classes):
    result_dict[i] = item


def imageClassPrediction(contents): 
    
    nparr = np.fromstring(contents, np.uint8)
    testing_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # testing_image = cv2.imread(file)


    gray = cv2.cvtColor(testing_image,cv2.COLOR_RGB2GRAY)

    faces = faceCascade.detectMultiScale(gray,1.1,4)

    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x: x+w]
        roi_color = testing_image[y:y+h, x: x+w]
        cv2.rectangle(testing_image, (x,y), (x+w,y+h), (255,0,0), 2)

        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Face not detected")
        else:
            for(ex,ey,ew,eh) in facess:
                face_roi = roi_color[ey: ey+eh, ex:ex+ew]

    final_image = cv2.resize(face_roi, (224,224))
    final_image = np.expand_dims(final_image,axis = 0)
    final_image = final_image/255.0

    Predictions = new_model.predict(final_image)
    return result_dict[np.argmax(Predictions)]

   



