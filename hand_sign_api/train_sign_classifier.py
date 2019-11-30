import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt


"""train_data = pd.read_csv('train/sign_mnist_train.csv')
test_data = pd.read_csv('test/sign_mnist_test.csv')
dataset = pd.concat([train_data, test_data], ignore_index=True)
dataset.to_csv(r'dataset.csv', index=False)"""

# Load dataset
dataset = pd.read_csv('data/dataset.csv')

# Create dictionary for alphabets and related numbers
alphabets_dic = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
             10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
             20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

alphabets = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# List of images
images = []

# List of labels
labels = []

for i in range(0, dataset.shape[0]):

    # Load row
    img_vector = dataset.iloc[i, 1:].values

    # Reshape into original dimension
    reshaped = np.reshape(img_vector, (-1, 28))

    # Convert type
    img = reshaped.astype(np.uint8)

    # Append to images list
    images.append(img)

    # Load label
    label_vector = dataset.iloc[i, 0]

    if label_vector > 25:
        print(label_vector)

    # Get related alphabet to digit
    #label = alphabets_dic[int(label_vector)]

    # Append to labels list
    labels.append([label_vector])

# Show label and image exple
"""x = 500
print(labels[x])
cv2.imshow('img', images[x])
cv2.waitKey(0)
cv2.destroyAllWindows()"""


labels_categories = []
for i in range(0,26):
    labels_categories.append([i])

# One hot encoding format for output
ohe = OneHotEncoder(handle_unknown='ignore', categorical_features=None)
ohe.fit(labels_categories)
labels = ohe.transform(labels).toarray()
data = np.array(images)
labels = np.array(labels)


# Split the data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=42)

print(X_train.shape)
print(X_test.shape)

X_train = X_train.reshape(27701,28,28,1)
X_test = X_test.reshape(6926,28,28,1)

print(X_train.shape)
print(X_test.shape)

# CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(26, activation='softmax'))

print(model.summary())
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=100, batch_size=64)

# Visualization
# Loss Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)

# Accuracy Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['acc'], 'r', linewidth=3.0)
plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)


plt.show()