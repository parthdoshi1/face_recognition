from pyexpat import model
import numpy as np
from PIL import Image
import os, cv2
import random
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
# (Remove this line entirely, as it's already imported at the top)
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from keras.layers import RandomFlip, RandomRotation, RandomZoom
import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator

def crop_variants(img, crop_size=(80, 80)):
    """Generate several crops: center, four corners, and a random crop."""
    w, h = img.size
    cw, ch = crop_size

    crops = []

    # Center crop
    left = (w - cw) // 2
    top = (h - ch) // 2
    crops.append(img.crop((left, top, left + cw, top + ch)))

    # Top-left
    crops.append(img.crop((0, 0, cw, ch)))
    # Top-right
    crops.append(img.crop((w - cw, 0, w, ch)))
    # Bottom-left
    crops.append(img.crop((0, h - ch, cw, h)))
    # Bottom-right
    crops.append(img.crop((w - cw, h - ch, w, h)))

    # Random crop
    if w > cw and h > ch:
        rand_left = random.randint(0, w - cw)
        rand_top = random.randint(0, h - ch)
        crops.append(img.crop((rand_left, rand_top, rand_left + cw, rand_top + ch)))

    return crops

def rotate_variants(img, angles=[-15, 0, 15]):
    """Return a list of images rotated by the given angles."""
    return [img.rotate(angle) for angle in angles]

def data_augmenter():
    '''
    Create a Sequential model composed of 2 layers
    Returns:
        tf.keras.Sequential
    '''
    ### START CODE HERE
    
    data_augmentation = keras.Sequential()
    data_augmentation.add(RandomFlip('horizontal'))
    data_augmentation.add(RandomRotation(0.2))
    ### END CODE HERE
    
    return data_augmentation

# Method to train custom classifier to recognize face
def train_classifer(name):
    # Read all the images in custom data-set
    path = os.path.join(os.getcwd()+"/data/"+name+"/")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = []
    ids = []
    labels = []
    pictures = {}

    # Store images in a numpy format and ids of the user on the same index in imageNp and id lists

    for root,dirs,files in os.walk(path):
            pictures = files


    for pic in pictures:
        imgpath = path + pic
        img = Image.open(imgpath).convert('L')
        img_np = np.array(img, 'uint8')
        id = int(pic.split(name)[0])
        faces.append(img_np)
        ids.append(id)
        # Detect faces in the image
        detected = face_cascade.detectMultiScale(img_np, scaleFactor=1.1, minNeighbors=5)
        angle_arr = []
        angle_arr.append([-5, 0, 5])  # Add the original image
        angle_arr.append([-10, 0, 10])
        angle_arr.append([-15, 0, 15])  
        angle_arr.append([-20, 0, 20])
        angle_arr.append([-25, 0, 25])
        angle_arr.append([-30, 0, 30])
        for angle in angle_arr:
            for rotated in rotate_variants(img, angles=angle):
                faces.append(np.array(rotated, 'uint8'))
                ids.append(id)     
    ids = np.array(ids)

    #Train and save classifier
    #clf = cv2.face.LBPHFaceRecognizer_create()
    #clf.train(faces, ids)
    #clf.write("./data/classifiers/"+name+"_classifier.xml")

    # CNN iplementation test
    IMAGE_SIZE = [224, 224]
    train_path = os.path.join(os.getcwd(), "data", "train")
    test_path = os.path.join(os.getcwd(), "data", "train")
    
    # Data augmentation
    data_augmentation = data_augmenter()
    
    # Data generators
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_input,
        validation_split=0.2  # 20% for validation
    )
    
    training_set = train_datagen.flow_from_directory(
        train_path,
        target_size=IMAGE_SIZE,
        batch_size=32,
        class_mode='categorical',
        subset='training',      # Set as training data
        shuffle=True
        )

    validation_set = train_datagen.flow_from_directory(
        train_path,
        target_size=IMAGE_SIZE,
        batch_size=32,
        class_mode='categorical',
        subset='validation',    # Set as validation data
        shuffle=True
    )
    
    vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    
    for layer in vgg.layers:
            layer.trainable = False
    
    x = vgg.output
    x = Flatten()(x)
    prediction = Dense(len(training_set.class_indices), activation='softmax')(x)
    
    model = Model(inputs=vgg.input, outputs=prediction)
    
    model.summary()
    
    model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
    )
    
    r = model.fit(
        training_set,
        validation_data=validation_set,
        epochs=5,
        steps_per_epoch=len(training_set),
        validation_steps=len(validation_set)
    )
    model.save('model.h5')
    import json
    with open('class_indices.json', 'w') as f:
        json.dump(training_set.class_indices, f)
    #train_classifer('tho1')