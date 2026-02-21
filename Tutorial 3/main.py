#Tahniat Khayyam 577608
#For Deep Learning Class (Spring 2026)
#%%


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers  import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

(X_train,y_train), (X_test,y_test)=mnist.load_data()
# Normalize 0 1
X_train=X_train.astype('float32')/255.0
X_test=X_test.astype('float32')/255.0

# One hot encoding
y_train=to_categorical(y_train,10) #0-9
y_test=to_categorical(y_test,10) #0-9

# building a NN model

model=Sequential([
    Flatten(input_shape=(28,28)),

    Dense(128,activation='relu'),
    Dense(64,activation='relu'),
    Dense(10,activation='softmax'),

    
])

model.summary()

# Compiling the mode
model.compile(optimizer=Adam(), loss='categorical_crossentropy',metrics=['accuracy'])
# training
history=model.fit(X_train,y_train, epochs=10,batch_size=32, validation_split=0.2)
# evaluation model

test_loss,test_accuracy=model.evaluate(X_test,y_test)
print(f"Test Acc: {test_accuracy:.4f}")


# Visualize Training
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],label="Train Accuracy")
plt.plot(history.history['val_accuracy'],label="Val Accuracy")
plt.title("Model accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# train and val loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'],label="Train Loss")
plt.plot(history.history['val_loss'],label="Val Loss")
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Predictions
predictions=model.predict(X_test)
plt.figure(figsize=(5,5))
plt.imshow(X_test[0],cmap='gray')
plt.title(f"True Label: {np.argmax(y_test[0])},Predicted: {np.argmax(predictions[0])}")
plt.axis('off')
plt.show()

num_images=9
plt.figure(figsize=(10,10))
for i in range(num_images):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[i],cmap='gray')
    plt.title(f"True: {np.argmax(y_test[i])},Predicted: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.tight_layout()
plt.show()