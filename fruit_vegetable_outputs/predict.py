
import tensorflow as tf
import numpy as np
import json

def load_model():
    model = tf.keras.models.load_model('fruit_vegetable_model.h5')
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    return model, class_indices

def predict_image(image_path):
    model, class_indices = load_model()
    
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    
    labels = {v: k for k, v in class_indices.items()}
    class_name = labels[predicted_class]
    
    return class_name, confidence
