import os
import cv2
from time import time
from PIL import Image
from tkinter import messagebox
from keras.models import load_model
import numpy as np
from keras.applications.vgg16 import preprocess_input
import json

def main_app(name, timeout = 15):
        
        face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
        model = load_model('./model.h5')
        with open('./data/classifiers/class.json', 'r') as f:
            class_indices = json.load(f)
        idx_to_class = {v: k for k, v in class_indices.items()}
        cap = cv2.VideoCapture(0)
        start_time = time()
        pred = False
    
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi_color = frame[y:y+h, x:x+w]
                # Preprocess for CNN
                face_img = cv2.resize(roi_color, (224, 224))
                face_array = np.expand_dims(face_img, axis=0)
                face_array = preprocess_input(face_array)
                # Predict
                preds = model.predict(face_array)
                class_id = np.argmax(preds)
                confidence = np.max(preds)
                label = idx_to_class[class_id]
                if confidence > 0.5:  # adjust threshold as needed
                    text = f"Recognized: {label} ({confidence:.2f})"
                    color = (0, 255, 0)
                else:
                    text = "Unknown Face"
                    color = (0, 0, 255)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                frame = cv2.putText(frame, text, (x, y-4), cv2.FONT_HERSHEY_PLAIN, 1, color, 1, cv2.LINE_AA)
            cv2.imshow("image", frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
            if time() - start_time > timeout:
                break
        cap.release()
        cv2.destroyAllWindows()

        '''
        if cv2.waitKey(20) & 0xFF == ord('q'):
            print(pred)
            if pred == True :
                
                dim =(124,124)
                img = cv2.imread(f".\\data\\{name}\\{pred}{name}.jpg", cv2.IMREAD_UNCHANGED)
                resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                cv2.imwrite(f".\\data\\{name}\\50{name}.jpg", resized)
                Image1 = Image.open(f".\\2.png") 
                  
                # make a copy the image so that the  
                # original image does not get affected 
                Image1copy = Image1.copy() 
                Image2 = Image.open(f".\\data\\{name}\\50{name}.jpg") 
                Image2copy = Image2.copy() 
                  
                # paste image giving dimensions 
                Image1copy.paste(Image2copy, (195, 114)) 
                  
                # save the image  
                Image1copy.save("end.png") 
                frame = cv2.imread("end.png", 1)
                cv2.imshow("Result",frame)
                cv2.waitKey(5000)
            
                messagebox.showinfo('Congrat', 'You have already checked in')
            else:
                messagebox.showerror('Alert', 'Please check in again')
            break
        '''
        elapsed_time = time() - start_time
        if elapsed_time >= timeout:
            print(pred)
            if pred:
                messagebox.showinfo('Congrat', 'You have already checked in')
            else:
                messagebox.showerror('Alert', 'Please check in again')


        
        
