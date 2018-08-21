
# coding: utf-8

# In[ ]:


**************** For Capturing Face************************************


# In[11]:



import cv2   
import numpy as np
import pickle

#Importing all the cascades files
face_cascade=cv2.CascadeClassifier('D:\\Mta DTU project\\Assignment\\3-Facial recognition\\src\\haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('D:\\Mta DTU project\\Assignment\\3-Facial recognition\\src\\haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('D:\\Mta DTU project\\Assignment\\3-Facial recognition\\src\\haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer.read("trainner.yml")
recognizer.read("D:\\Mta DTU project\\Assignment\\3-Facial recognition\\images\\trainner.yml")

labels = {"person_name": 1}

with open("labels.pickle",'rb')as f: #for label id
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
    #label=pickle.load(f)
    
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    
    for (x, y, w, h) in faces:
        #print(x,y,w,h)
        
        roi_gray = gray[y:y+h, x:x+w]#For gray
        roi_color = frame[y:y+h, x:x+w]
        
        #recognise using deep learned model keras these are complicated
        
        id_,conf=recognizer.predict(roi_gray)
        
        if conf>=45 and conf<=85:
            print(id_)
            print(label[id_])
            
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            stroke=2
            
             
#cv2 put text for puttting image on image
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        
        img_item="D:\\Mta DTU project\\Assignment\\3-Facial recognition\\images\\output image\\my-image.png"
        
        cv2.imwrite(img_item,roi_color)#for writing image to file
       # cv2.imwrite(img_item,roi_gray)
        
        color =(255,0,0)
        stroke=2
        end_cord_x=x+w
        end_cord_y=y+h
        
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)
        
        #for eyes and smile facial recognition
        
        eyes=eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
            
        subitems=smile_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in subitems:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        
        # Display the resulting frame
    cv2.imshow('frame',frame)
        
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break   
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# In[ ]:


***************************** Training Data *******************************************************


# In[2]:


import cv2   
import os
import numpy as np
from PIL import Image
import pickle 

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))#to find the dir of this  file
image_dir = os.path.join(BASE_DIR, "D:\\Mta DTU project\\Assignment\\3-Facial recognition\\images")#to find dir of image

face_cascade=cv2.CascadeClassifier('D:\\Mta DTU project\\Assignment\\3-Facial recognition\\src\\haarcascade_frontalface_alt2.xml')

#train the opencv recogniser #use lbphf recognizer

recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.face.createLBPHFaceRecognizer()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):#to print the images and the dir
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            
            #to get the path of the image
            path = os.path.join(root, file)
            
            #to get the name of the folder 
            #replace is used here to replace any space with dash and lower will lower case it
            label = os.path.basename(root).replace(" ", "-").lower()
         
            
            #.......for getting label id.......#
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            
            id_ = label_ids[label]
            
           # print(label_ids)
            
            
            #y_train.append(label)
            #x_train.append(path)
            
            
            #we want to use number value for labels and convert into numpy array and also convert into gray
            
            #Training image into array
            
            pil_image = Image.open(path).convert("L") # converts it into grayscale
            # image_array = np.array(pil_image, "uint8")#converting grayscale into numpy array
            
            #resizing the image
            size = (550, 550)
            
            
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            
           
            
            image_array = np.array(final_image, "uint8")
            
            #print(image_array)#printing the image into the form of numpy array by taking pixel value
            
            #for finding region of interest #face detection
            
            faces=face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)
            
            for (x, y, w, h) in faces:
                
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
                
                #.... creating Training labels... #
                
    
    #print(x_train)
    #print(y_labels)
            
  #labels are needed for prediction          
with open("labels.pickle",'wb')as f:
    pickle.dump(label_ids,f)
    

 #recognizer   
recognizer.train(x_train, np.array(y_labels))

#recognizer.save("recognizers/face-trainner.yml")
recognizer.save("D:\\Mta DTU project\\Assignment\\3-Facial recognition\\images\\trainner.yml")

