import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

input_size = (48, 48, 3) 
timesteps = 10  
num_classes = 3  
num_epochs = 10    
learning_rate = 0.001    

def load_data(data_dir):
    X, y = [], []
    class_labels = os.listdir(data_dir)  
    
    # Inisialisasi ImageDataGenerator untuk augmentasi
    datagen = ImageDataGenerator(rescale=1./255)
    
    for label in class_labels:
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        
        # Proses tiap folder (label)
        image_paths = [os.path.join(label_dir, img) for img in os.listdir(label_dir)]
        
        for i in range(0, len(image_paths) - timesteps, timesteps):
            sequence = []
            for j in range(timesteps):
                img = load_img(image_paths[i + j], target_size=input_size[:2])
                img = img_to_array(img)
                sequence.append(img)
            
            X.append(sequence)
            y.append(label)    
    
    # Konversi ke numpy array
    X = np.array(X)
    y = np.array(y)
    
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    
    return X, y

X_train, y_train = load_data('../data/train')
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Membangun Model RNN dengan LSTM
model = models.Sequential([
    layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'), input_shape=(timesteps, *input_size)),
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')),
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    
    layers.TimeDistributed(layers.Flatten()),
    layers.LSTM(64, return_sequences=False),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val))

model.save('models/rnn_model.h5')
print("Model telah disimpan di 'models/rnn_model.h5'.")
