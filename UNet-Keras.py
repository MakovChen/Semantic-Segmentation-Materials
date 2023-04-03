import tensorflow_datasets as tfds
from keras import backend as K
import tensorflow as tf
import numpy as np
import cv2, keras
import matplotlib.pyplot as plt

# Load and Preprocesing to the Oxford-IIIT Pet Dataset from TensorFlow
dataset = tfds.load('oxford_iiit_pet:3.2.0')
images, labels = [], []

def resize(input_image, input_label):
    input_image = cv2.resize(np.array(input_image), (128, 128), cv2.INTER_AREA)
    input_label = cv2.resize(np.array(input_label), (128, 128), cv2.INTER_AREA)
    return input_image, input_label 

def augment(input_image, input_label):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_label = tf.image.flip_left_right(input_label)
    return input_image, input_label

def normalize(input_image, input_label):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_label -= 1
    return input_image, input_label

for sample in dataset['train']:
    image, label = resize(sample['image'], sample['segmentation_mask'])
    images.append(image); labels.append(label) 
images, labels = np.array(images), np.array(labels)
X, y = normalize(images, labels)
X, y = augment(X, y)
print('images shape:', X.shape)
print('labels shape:', y.shape)

'''
[Build the Unet model]
'''
def double_conv_block(x, n_filters):
    x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    return x

def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = tf.keras.layers.MaxPool2D(2)(f)
    p = tf.keras.layers.Dropout(0.3)(p)
    return f, p

def upsample_block(x, conv_features, n_filters):
    x = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = tf.keras.layers.concatenate([x, conv_features])
    x = tf.keras.layers.Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)
    return x

def UNet(inputs):
    # encoder: contracting path - downsample
    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)

    # decoder: expanding path - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)

    outputs = tf.keras.layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)
    model = tf.keras.Model(inputs, outputs, name="U-Net")
    return model
    
def segmentation_model():
    num_classes = len(np.unique(y)) #0, 1, 2
    inputs_shape = (128, 128, 3)
    inputs = tf.keras.layers.Input(shape = inputs_shape)
    model = UNet(inputs)
    model.summary()
    return model

model = segmentation_model()
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(X[:-1], y[:-1], batch_size=32, epochs=10)

'''
[Utilize AND Demo]
'''
def display(display_list):
  plt.figure(figsize=(15, 15))
  title = ["Input Image", "True Mask", "Predicted Mask"]
  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis("off")
    plt.show()
pred = model.predict(X[-1:])
display([X[-1], np.array(y[-1]).reshape(128, 128, 1), np.argmax(pred[-1], axis=2).reshape(128, 128, 1)])
