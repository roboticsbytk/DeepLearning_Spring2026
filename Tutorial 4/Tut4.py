import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
import os
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers  import Dense, Flatten

# folder
save_folder=r"D:\PHD PROGRAMME\Courses\Sem2\DeepLearning\Tuts\T4\Augmented Images"
os.makedirs(save_folder,exist_ok=True)

# parameters for ImageDataGen--> Augmentation

datagen=ImageDataGenerator(
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.5,1.5)
)

# model=Sequential([
#     Flatten(input_shape=(28,28)),
#     Dense(128,activation='relu'),
#     Dense(64,activation='relu'),
#     Dense(10, activation='softmax')

# ])

# model.summary()

# load an image
img=load_img(r"D:\PHD PROGRAMME\Courses\Sem2\DeepLearning\Tuts\T4\images.jpg")
x=img_to_array(img)
# Reshape
x=x.reshape((1,)+x.shape)
# generating aug. images

i=0
for batch in datagen.flow(x,batch_size=1,
                          save_to_dir=save_folder,
                          save_prefix='image',
                          save_format='jpeg'
                          ):
    augmented_image=batch[0].astype('uint8')

    plt.figure()
    plt.imshow(augmented_image)
    plt.axis('off')
    plt.show()

    i+=1

    if i>20:
        break

print(f"Augmented Images are saved in the folder: {os.path.abspath(save_folder)}")