# Overfitting is caused by having too few samples to learn from, rendering us unable to train a model able to generalize to new data
# Data augmentation takes the approach of generating more training data from existing training samples, by "augmenting" the samples via a number of random transformations that yield believable-looking images.
# Better generalization

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
import os
import matplotlib.pyplot as plt

# use this to implement data augmentation
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# rotation_range : 隨意旋轉影像的角度（0-180）
# width_shift_range / height_shift_range : 平移影像
# shearing_range : 隨機順時針地傾斜影像
# zoom_range : 隨機shrink/increase影像
# horizontal_flip : 隨機反轉一半影像
# fill_mode : 新建影像的時候填補像素的方法

base_dir = r'C:\Users\User\PycharmProjects\pythonProject\NTU2021_ML\CNN_demo'
train_dir = os.path.join(base_dir, 'train')
train_cats_dir = os.path.join(train_dir, 'cats')
fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]


# We pick one image to "augment"
img_path = fnames[3]

# Read the image and resize it
img = image.load_img(img_path, target_size=(150, 150))

# Convert it to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)

# Reshape it to (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)

# The .flow() command below generates batches of randomly transformed images.
# It will loop indefinitely, so we need to `break` the loop at some point!
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()