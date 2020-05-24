import keras
from keras import layers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.utils.data_utils import get_file
import numpy as np

def load_mnist():
    """ Chargement et pré-processing de MNIST.
    MNIST est un dataset contenant 70.000 images de chiffres manuscrits.
    Cette fonction charge le dataset à partir d'AWS, et le prépare pour 
    l'entrainement d'un modèle.
    :output : les bases d'apprentissage et de test, sous la forme d'arrays numpy.
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

    # Reshaping des données
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
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
    return (X_train, y_train), (X_test, y_test)


def build_model(X_train, y_train, X_test, y_test, batch_size = 128, epochs = 1):
    """Crée, entraine et retourne un CNN pour la prédiction de chiffres manuscrits.
    L'architecture du réseau consiste en deux couches de convolution, suivies d'une couche de 
    pooling et de deux couches denses, entrecoupées de dropout. 
    L'output du réseau est un array sparse avec un seul 1, dont l'index est la classe prédite.
    :output : le modèle entrainé.
    """
    model = Sequential([
        layers.Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(28,28,1)),
        layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy']
    )

    model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=1,
        verbose=1,
        validation_data=(X_test, y_test)
    )

    return model


if __name__=="__main__":
    (X_train, y_train), (X_test, y_test) = load_mnist()
    model = build_model(X_train, y_train, X_test, y_test)
    # Evaluation du modèle
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Sauvegarde du modèle
    model.save("model/model.h5")
    print("Modèle sauvegardé sous model/model.h5")
    