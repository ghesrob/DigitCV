from argparse import ArgumentParser

import numpy as np
from tensorflow.keras import layers, backend
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, get_file


def load_mnist():
    """Loading and preprocessing of the MNIST Dataset.
    MNIST is a large dataset of handwritten digits.
    This function load MNIST from AWS, and preprocess it for training a model.
    """
    # Dataset loading
    path = get_file(
        fname="mnist.npz",
        origin='https://s3.amazonaws.com/img-datasets/mnist.npz',
        file_hash='8a61469f7ea1b51cbae51d4f78837e45'
    )
    with np.load(path, allow_pickle=True) as file:
        X_train, y_train = file['x_train'], file['y_train']
        X_test, y_test = file['x_test'], file['y_test']

    # Data reshaping according to user's system image data format
    if backend.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
        input_shape = (1, 28, 28)
    else:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)

    # Normalization
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255
    #X_train = (X_train > 0).astype("int")
    #X_test = (X_test > 0).astype("int")
    # One-hot-encoding 
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    # Output
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    return (X_train, y_train), (X_test, y_test), input_shape


def create_model(input_shape):
    """Create a convolutional neural network and an image data generator.
    Build and compile a sequential CNN for handwritten digits recognition. The returned CNN is ready for training.
    Also create an image data generator to train the model on.
    """
    # Model building
    model = Sequential([
        layers.Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=input_shape),
        layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])
    # Model compilation
    model.compile(
        loss=categorical_crossentropy,
        optimizer=Adadelta(),
        metrics=['accuracy']
    )
    # Image data generator
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.1,
        fill_mode='nearest'
    )

    return (model, datagen)


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument("--name", type=str, default="model")
    options = parser.parse_args()

    # Data loading and model creation
    (X_train, y_train), (X_test, y_test), input_shape = load_mnist()
    model, datagen = create_model(input_shape)

    # Model training, using augmented data
    model.fit(
        datagen.flow(X_train, y_train, batch_size=options.batch_size),
        epochs=options.epochs,
        verbose=1,
        validation_data=(X_test, y_test),
        steps_per_epoch=X_train.shape[0] // options.batch_size
    )

    # Evaluation and saving as .h5 file
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    path = f"models/{options.name}.h5"
    model.save(path)
    print("Model saved as " + path)
    
