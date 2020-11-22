import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow import keras




class_names = ['feuille','pierre','ciseaux']


model = keras.models.load_model('CHEMIN DU MODEL/sequential_model')


cap = cv.VideoCapture(0)

while(True):
    # Capture image par image
    ret, img = cap.read()
    img = cv.flip(img,1)
    cv.rectangle(img, (375,75), (600,300), (0,255,0))
    rect = img[75+2:300-2,375+2:600-2]
 
    #Redimmension de l'image  
    rect = keras.preprocessing.image.smart_resize(
        rect, (64,64), interpolation='bilinear'
    )
    img_array = keras.preprocessing.image.img_to_array(rect)
    img_array = tf.expand_dims(img_array, 0) 
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])    

    cv.putText(img,class_names[np.argmax(score)],(70,170),cv.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),2)
    # Préparation de l'affichage de l'image
    cv.imshow('frame',img)
            
    # affichage et saisie d'un code clavier
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# Ne pas oublier de fermer le flux et la fenetre
cap.release()
cv.destroyAllWindows()
