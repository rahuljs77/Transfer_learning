from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input

#  Transfer learning

freeze_flag = True  # `True` to freeze layers, `False` for full training
weights_flag = 'imagenet'  # 'imagenet' or None
preprocess_flag = True  # Should be true for ImageNet pre-trained typically

# Loads in InceptionV3
from tensorflow.keras.applications.inception_v3 import InceptionV3

input_size = 139

# Using Inception with ImageNet pre-trained weights
inception = InceptionV3(weights=weights_flag, include_top=False,
                        input_shape=(input_size, input_size, 3))

if freeze_flag:
    for layer in inception.layers:
        layer.trainable = False


cifar_input = Input(shape=(32, 32, 3))
resized_input = Lambda(lambda image: tf.image.resize(image, (input_size, input_size)))(cifar_input)
inp = inception(resized_input)
x = GlobalAveragePooling2D()(inp)
x = Dense(512, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=cifar_input, outputs=predictions)
model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])

(X_train, y_train), (X_val, y_val) = cifar10.load_data()

# One-hot encode the labels
label_binarizer = LabelBinarizer()
y_one_hot_train = label_binarizer.fit_transform(y_train)
y_one_hot_val = label_binarizer.fit_transform(y_val)

# Shuffle the training & test data
X_train, y_one_hot_train = shuffle(X_train, y_one_hot_train)
X_val, y_one_hot_val = shuffle(X_val, y_one_hot_val)

# We are only going to use the first 10,000 images for speed reasons
# And only the first 2,000 images from the test set
X_train = X_train[:10000]
y_one_hot_train = y_one_hot_train[:10000]
X_val = X_val[:2000]
y_one_hot_val = y_one_hot_val[:2000]

if preprocess_flag:
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
else:
    datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()

batch_size = 32
epochs = 5
# Note: we aren't using callbacks here since we only are using 5 epochs to conserve GPU time
model.fit_generator(datagen.flow(X_train, y_one_hot_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train)/batch_size, epochs=epochs, verbose=1,
                    validation_data=val_datagen.flow(X_val, y_one_hot_val, batch_size=batch_size),
                    validation_steps=len(X_val)/batch_size)
# model.fit(X_train, y_one_hot_train, validation_split = 0.2, batch_size = batch_size, epochs = epochs, verbose=1)

