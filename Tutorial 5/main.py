import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

(train_images,train_labels),(test_images,test_labels)=tf.keras.datasets.cifar10.load_data()

train_images=train_images/255.0
test_images=test_images/255.0

class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# visualize

plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(train_images[i])
    plt.title(class_names[train_labels[i][0]])
    plt.axis('off')
plt.show()


# model: CNN 

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

history = model.fit(train_images,train_labels,    epochs=10,    validation_data=(test_images,test_labels))

test_loss, test_acc = model.evaluate(test_images,test_labels, verbose=2)

print("Test accuracy:", test_acc)

plt.figure(figsize=(12,4))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.title('Training and Val accuracy')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Val loss')
plt.legend()

plt.show()


predictions = model.predict(test_images[:5])

import numpy as np
import matplotlib.pyplot as plt

def plot_image(i, predictions_array, true_label, img, class_names):

    predictions_array,true_labe,img = predictions_array[i],true_label[i][0],img[i]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)

    color = 'blue' if predicted_label == true_label else 'red'


    plt.xlabel(f"Predicted: {class_names[predicted_label]} (True: {class_names[true_label[0]]})",color=color)

plt.figure(figsize=(5,10))
for i in range(5):
    plt.subplot(5,1,i+1)
    plot_image(i,predictions,test_labels,test_images,class_names)

plt.tight_layout()
plt.show()