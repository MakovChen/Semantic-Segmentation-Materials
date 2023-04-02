import os
import cv2
import numpy as np
import tensorflow as tf

#Download Dataset and Preprocessing
def preprocess_data(data_path, img_size):
    # Get the paths to the images and their corresponding masks
    img_paths = sorted(os.listdir(os.path.join(data_path, "images")))
    mask_paths = sorted(os.listdir(os.path.join(data_path, "annotations", "trimaps")))

    # Initialize arrays to store the images and masks
    X = np.zeros((len(img_paths), img_size, img_size, 3), dtype=np.float32)
    y = np.zeros((len(mask_paths), img_size, img_size, 1), dtype=np.float32)

    # Loop over the images and masks and preprocess them
    for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
        # Load the image and mask
        img = cv2.imread(os.path.join(data_path, "images", img_path))
        mask = cv2.imread(os.path.join(data_path, "annotations", "trimaps", mask_path))

        # Resize the image and mask
        img = cv2.resize(img, (img_size, img_size))
        mask = cv2.resize(mask, (img_size, img_size))

        # Normalize the pixel values of the image
        img = img / 255.0

        # Convert the mask to a binary image
        mask = mask[:, :, 0:1] // 128
        mask = np.float32(mask)

        # Store the image and mask in the arrays
        X[i] = img
        y[i] = mask

    return X, y

data_path = "./"
img_size = 256
X, y = preprocess_data(data_path, img_size)
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_val, y_val = X[split:], y[split:]

#Build FCN
def FCN(num_classes, input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(pool3)
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up1)
    up2 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv4)
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up2)
    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax')(conv5)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
    
model = FCN(num_classes=3, input_shape=(img_size, img_size, 3))
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=16, epochs=50, validation_data=(X_val, y_val))

#Demo
img = cv2.imread("./image.jpg")
img = cv2.resize(img, (img_size, img_size))
img = img / 255.0

pred = model.predict(np.array([img]))
pred = np.argmax(pred, axis=-1)
pred = np.uint8(pred[0])

cv2.imshow("Prediction", pred * 128)
cv2.waitKey(0)
cv2.destroyAllWindows()
