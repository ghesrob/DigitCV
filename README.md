# DigitsCV

A computer vision application that uses convolutional neural network for handwritten digits recognition. Neural networks are powered by TensorFlow and trained on the MNIST dataset.

# Installation

Requiere Python version 3.5-3.7 for [compatibility with TensorFlow](https://www.tensorflow.org/install/pip?hl=fr)

```
git clone https://github.com/ghesrob/Digits-CV.git
cd Digits-CV
python -m venv venv
venv/scrits/activate.ps1
pip install requirements.txt
```

```
git clone https://github.com/ghesrob/Digits-CV.git
cd Digits-CV
python3 -m venv venv
source 
pip3 install requirements.txt
```

Execute `python gui.py` (`python3 gui.py` for linux-users) to start the GUI. Draw some digits on the dedicated area, and press "predict" to let the CNN recognize your handwriting.

