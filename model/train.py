import keras
from keras import layers, backend
from keras.models import Sequential
from keras.utils import to_categorical
from keras.utils.data_utils import get_file
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def load_mnist():
    """ Chargement et pré-processing de MNIST.
    MNIST est un dataset contenant 70.000 images de chiffres manuscrits.
    Cette fonction charge le dataset à partir d'AWS, et le prépare pour 
    l'entrainement d'un modèle.

    :output : les bases d'apprentissage et de test sous la forme d'arrays numpy, ainsi
    que les dimensions d'une image.
    """
    # Chargement du dataset
    path = get_file(
        fname="mnist.npz",
        origin='https://s3.amazonaws.com/img-datasets/mnist.npz',
        file_hash='8a61469f7ea1b51cbae51d4f78837e45'
    )
    with np.load(path, allow_pickle=True) as file:
        X_train, y_train = file['x_train'], file['y_train']
        X_test, y_test = file['x_test'], file['y_test']

    # Reshaping des données selon le format d'image du système
    if backend.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
        input_shape = (1, 28, 28)
    else:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)

    # Normalisation des données
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # One-hot-encoding des classes
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Sortie
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    return (X_train, y_train), (X_test, y_test), input_shape


def build_model(X_train, y_train, X_test, y_test, input_shape, batch_size = 128, epochs = 1):
    """Crée, entraine et retourne un CNN pour la prédiction de chiffres manuscrits.
    L'architecture du réseau consiste en deux couches de convolution, suivies d'une couche de 
    pooling et de deux couches denses, entrecoupées de dropout. 
    L'output du réseau est un array sparse avec un seul 1, dont l'index est la classe prédite.

    :output : le modèle entrainé.
    """
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

    # Augmentation de données
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.1,
        fill_mode='nearest'
    )

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy']
    )

    # Entrainement du modèle sur données augmentées
    model.fit_generator(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, y_test),
        steps_per_epoch=X_train.shape[0] // batch_size
    )

    return model


if __name__=="__main__":
    (X_train, y_train), (X_test, y_test), input_shape = load_mnist()
    model = build_model(X_train, y_train, X_test, y_test, input_shape)
    # Evaluation du modèle
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Sauvegarde du modèle
    model.save("model/model.h5")
    print("Modèle sauvegardé sous model/model.h5")
    
