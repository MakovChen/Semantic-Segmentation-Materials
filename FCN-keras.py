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
[Build the FCN model]

'''
def VGG16(inputs_shape):
    #as the encoder in FCN
    base_model = keras.applications.vgg16.VGG16(include_top=False, input_shape=inputs_shape)
    layer_names = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool',]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    base_model.trainable = False
    return tf.keras.models.Model(base_model.input, base_model_outputs)

def FCN(convs, num_classes):
      func1, func2, func3, func4, func5 = convs
      #as the decoder in FCN
      conv6 = tf.keras.layers.Conv2D(4096, (7, 7) , activation='relu', padding='same',name="conv6")(func5)
      conv7 = tf.keras.layers.Conv2D(4096, (1, 1) , activation='relu', padding='same',name="conv7")(conv6)
      out = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2) ,use_bias=False)(conv7)
      out = tf.keras.layers.Cropping2D(cropping=(1, 1))(out)
      out2 = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='relu', padding='same')(func4)
      out = tf.keras.layers.Add()([out, out2])
      out = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(out)
      out = tf.keras.layers.Cropping2D(cropping=(1, 1))(out)
      out3 = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='relu', padding='same')(func3)
      out = tf.keras.layers.Add()([out, out3])
      out = tf.keras.layers.Conv2DTranspose(num_classes , kernel_size=(8, 8) ,  strides=(8, 8) , use_bias=False)(out)
      return tf.keras.layers.Activation('softmax')(out)
    
def segmentation_model():
    num_classes = len(np.unique(y)) #0, 1, 2
    inputs_shape = (128, 128, 3)
    inputs = keras.layers.Input(shape = inputs_shape)
    Encoder = VGG16(inputs_shape)(inputs)
    Decoder = FCN(Encoder, 3)
    model = tf.keras.Model(inputs=inputs, outputs=Decoder)
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
