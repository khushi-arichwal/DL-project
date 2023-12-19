# Importing libraries 
from tensorflow.keras import Sequential, datasets
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
from matplotlib import pyplot as plt 
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Loading datasets
(X_train, Y_train), (X_test, Y_test) = datasets.cifar10.load_data()

# reshape
Y_train = Y_train.reshape(-1, )

Y_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
Y_train

# observe dataset
def showImage(x,y,index):
    plt.figure(figsize=(15,2))
    plt.imshow(x[index])
    plt.xlabel(Y_classes[y[index]])
    
showImage(X_train,Y_train,8)
    
# Normalize
X_train = X_train/255
X_test = X_test/255

X_train[0]

# Build model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size = (3,3), activation='relu', input_shape = (32,32,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
          
model.add(Conv2D(filters=64, kernel_size = (4,4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
          
model.add(Flatten())
model.add(Dense(units=32, activation='tanh'))
model.add(Dense(units=10, activation='softmax'))

# compile model
model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

# train model
model.fit(X_train, Y_train, validation_data = (X_test, Y_test) , epochs= 6)

# predictions
Y_predictions = model.predict(X_test)
Y_predictions[9]

Y_predictions = [np.argmax(arr) for arr in Y_predictions]
Y_predictions

Y_test = Y_test.reshape(-1, )
Y_predictions[9]

showImage(X_test, Y_test, 1 )

# evaluate model
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_predictions))
