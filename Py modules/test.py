from tensorflow import keras
from tensorflow.keras import Model

from keras.models import load_model
new_model = load_model('model_weight\cat_vs_dog_model.h5')

from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def checkModel(path):
    sample = path
    img_pred=image.load_img(sample,target_size=(224,224))
    load_img=image.load_img(sample)

    img_pred=image.img_to_array(img_pred)
    img_pred=np.expand_dims(img_pred, axis=0)
    prediction = ""
    rslt= new_model.predict(img_pred)

    print(rslt)
    if rslt[0][0]>rslt[0][1]:
        prediction="cat"
        
        plt.imshow(load_img)
        plt.show()
        
    else:
        prediction="dog"
        plt.imshow(load_img)
        plt.show()
        
    print(prediction)