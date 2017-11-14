from keras.datasets import cifar10
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
#from tensorflow.contrib.keras.python.keras.utils.np_utils import to_categorical
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


def plot_image(X, label=None):
    print('☆テスト画像: %s' % label)
    plt.imshow(X)
    plt.show()
    plt.clf()

if __name__=='__main__':
    ## import data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.reshape([-1, 32, 32, 3])
    X_test = X_test.reshape([-1, 32, 32, 3])
    print('%i training samples' % X_train.shape[0])
    print('%i test samples' % X_test.shape[0])
    print(X_train.shape)

    # convert integer RGB values (0-255) to float values (0-1)
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    ## CIFAR-10公式ラベル
    cifar10_labels = np.array([
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'])

    # create model
    model = load_model('my_model.h5') # 保存したモデルから読み込む

    # output
    preds = model.predict(X_test) # modelは学習させたもの
    category = np.argmax(preds, axis=1)
    for i in range(10):
        classed = category[i]
        labels =  flower_vgg_label[classed]
        plot_image(X_test[i], labels)
        print('%i - 分類されたものは%sです。精度は%f％です。\n' % (i, labels, preds[i][classed]*100))
        print("################################################")
