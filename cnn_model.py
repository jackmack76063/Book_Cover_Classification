# Name: Mackenzie Jackson
# Class: CS 435: Deep Learning
# Final Project: Project CNN Model
# Description: This program will train book cover images along with their 
# corresponding genre category. It will then use this to predict the literary
# genres of new book cover images. 
# Date: 05/29/26
# Citations: 
# Source 1: CS 435 CNN Explorations
# Source URL: https://canvas.oregonstate.edu/courses/2055741/pages/exploration-programming-cnns-for-image-classification-in-python-2?module_item_id=26517829
# Source 2: Python Image Processing
# Source URL: https://note.nkmk.me/en/python-numpy-image-processing/
# Source 3: Resizing an image
# Source URL: https://numpy.org/devdocs/reference/generated/numpy.resize.html
# Source 4: Converting NP array to smaller data type
# Source URL: https://www.zerve.ai/data-science-problems/numpy/memoryerror-unable-to-allocate-array-fix
# Source5: Using dropout to prevent overfitting
# Source URL: http://geeksforgeeks.org/deep-learning/tf-keras-layers-dropout-in-tensorflow
# Source 6: ImageDataGenerator()
# Source URL: https://medium.com/@bcwalraven/boost-your-cnn-with-the-keras-imagedatagenerator-99b1ef262f47

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns


#########################################################
## Prepare the Data for a CNN Model 
#########################################################
bookData = pd.read_csv("book_dataset_final.csv")

"""
selected_categories = [
    "Mystery, Thriller & Suspense",
    "Comics & Graphic Novels",
    "Cookbooks, Food & Wine",
    "Romance",
    "Travel"
]

bookData = bookData[bookData["Category"].isin(selected_categories)]
"""

print("\nOriginal Dataset Shape:", bookData.shape)
print("\nOriginal Category Counts:")
print(bookData["Category"].value_counts())


#Split data into training/testing groups
train_df, test_df = train_test_split(
    bookData,
    test_size=0.2,
    random_state=42,
    stratify=bookData["Category"])

#Confirm size of groups after split
print("\nTraining shape: ", train_df.shape)
print("\nTesting shape: ", test_df.shape)

#########################################################
## Utilize Image Generators 
#########################################################

#image generators will assist in loading, rescaling, reformatting, resizing

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col = "Filepath",
    y_col = "Category",
    target_size=(240,240),
    color_mode="rgb",
    batch_size=32,
    class_mode="sparse",
    shuffle=True
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="Filepath",
    y_col="Category",
    target_size=(240, 240),
    color_mode="rgb",
    batch_size=32,
    class_mode="sparse",
    shuffle=False
)



#########################################################
## Build the CNN Model 
#########################################################



CNN_Model = tf.keras.models.Sequential([
 tf.keras.layers.Conv2D(input_shape=(240,240,3), kernel_size=(3,3), filters=32, activation="relu"), 
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
 
 tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), 
 
 tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
 tf.keras.layers.MaxPooling2D((2,2)),
 
 tf.keras.layers.Flatten(), 
 tf.keras.layers.Dense(64, activation='relu'), 
 tf.keras.layers.Dropout(0.3),
 
 tf.keras.layers.Dense(5, activation='softmax') 
])

CNN_Model.summary()

#compile model
CNN_Model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

##Set epochs for accuracy improvement
history = CNN_Model.fit(train_generator, epochs=20, 
                    validation_data=test_generator)

## Plot Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

## Test the Model ---------------------------------------------------------
test_loss, test_acc = CNN_Model.evaluate(test_generator, verbose=2)
print("\nTest Accuracy: ")
print(test_acc)

## Make Predictions
CNNpredictions=CNN_Model.predict(test_generator)
print("\nCNN Predictions: ")
print(CNNpredictions)
print("\nCNN Shape: ")
print(CNNpredictions.shape)

#########################################################
## Confusion Matrix
#########################################################
Pred_Max_Values = np.argmax(CNNpredictions, axis=1)
test_labels = test_generator.classes
CNN_CM=confusion_matrix(test_labels, Pred_Max_Values)

class_names = list(test_generator.class_indices.keys())

fig, ax = plt.subplots(figsize=(15,15)) 
sns.heatmap(CNN_CM, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})

ax.set_xlabel('Predicted Genres') 
ax.set_ylabel('True Genres')
ax.set_title('Confusion Matrix:') 
ax.xaxis.set_ticklabels(class_names, rotation=90, fontsize=18)
ax.yaxis.set_ticklabels(class_names, rotation=0, fontsize=18)

plt.show()
