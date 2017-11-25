import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Dense, LSTM
# from keras.layers.embeddings import Embedding
# from keras.preprocessing import sequence
from keras.callbacks import History
# from keras.models import model_from_json
from keras.utils import plot_model
from IPython.display import SVG
# from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
try:
    # pydot-ng is a fork of pydot that is better maintained
    import pydot_ng as pydot
except ImportError:
    # fall back on pydot if necessary
    import pydot
from PIL import Image
import csv
from numpy.random import *
from operator import itemgetter

image_size = 50
history = History()

# with open('lenet.json', 'r') as file:
#     model_json = file.read()

# if os.path.getsize(history) > 0 :
# with open(history, 'rb') as f:
#     history = pickle.load(f)

f = open('images/metadata/train_metadata.csv', 'rt')
train_dataReader = csv.reader(f)
train_data = [ e for e in train_dataReader]
f.close()

g = open('images/metadata/test_metadata.csv', 'rt')
test_dataReader = csv.reader(g)
test_data = [ e for e in test_dataReader]
g.close()

train_num = 0
test_num = 0


error=[]
prediction=[]

# 学習用のデータを作る.
image_list = []
label_list = []
campus_list = []
campus_name = []

# ./data/train 以下のorange,appleディレクトリ以下の画像を読み込む。
for file in os.listdir("images/all"):
    if file != ".DS_Store":
        for i in train_data:
            if i[0]==file:
                label = i[1]
                del i
                break
        # 配列label_listに正解ラベルを追加(りんご:0 オレンジ:1)
        label_list.append(label)
        filepath = "images/all" + "/" + file
        # 画像を25x25pixelに変換し、1要素が[R,G,B]3要素を含む配列の25x25の２次元配列として読み込む。
        # [R,G,B]はそれぞれが0-255の配列。
        image = randint(255, size =(image_size,image_size,3))
        image_ = np.array(Image.open(filepath))
        image_ = np.array(Image.open(filepath).resize((image_size, image_size)))
        if image_.shape[2]==4:
            for j in range(image_size):
               for k in range(image_size):
                    image[j][k] = image_[j][k][0:3]
            # print(image.shape)
        # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
        # image = image.transpose(2, 0, 1)
        # さらにフラットな1次元配列に変換。最初の1/3はRed、次がGreenの、最後がBlueの要素がフラットに並ぶ。
        # image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
        # print(image.shape)
        # 出来上がった配列をimage_listに追加。
        image_list.append(image)

for file in os.listdir("images/campus"):
    if file != ".DS_Store":
        # 配列label_listに正解ラベルを追加(りんご:0 オレンジ:1)
        filepath = "images/campus" + "/" + file
        # 画像を25x25pixelに変換し、1要素が[R,G,B]3要素を含む配列の25x25の２次元配列として読み込む。
        # [R,G,B]はそれぞれが0-255の配列。
        image = randint(255, size =(image_size,image_size,3))
        image_ = np.array(Image.open(filepath))
        image_ = np.array(Image.open(filepath).resize((image_size, image_size)))
        print(file)
        print(image_[0])
        if image_.shape[2]==4:
            for j in range(image_size):
               for k in range(image_size):
                    image[j][k] = image_[j][k][0:3]
            # print(image.shape)
        # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
        # image = image.transpose(2, 0, 1)
        # さらにフラットな1次元配列に変換。最初の1/3はRed、次がGreenの、最後がBlueの要素がフラットに並ぶ。
        # image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
        # print(image.shape)
        # 出来上がった配列をimage_listに追加。
        campus_list.append(image)

# # kerasに渡すためにnumpy配列に変換。
# print(len(image_list[0]))
image_list = np.array(image_list)
campus_list = np.array(campus_list)
image_list = image_list.astype("float32")
campus_list = campus_list.astype("float32")
print(image_list[0])
image_list = image_list / 255 # for n in image_list:
campus_list = campus_list / 255
    # print(n.shape)
#
# ラベルの配列を1と0からなるラベル配列に変更
# 0 -> [1,0], 1 -> [0,1] という感じ。
# Y = to_categorical(label_list)
Y = np.array(label_list)
Y = Y.astype("float32")
#divide train and test data
X_train, X_test, y_train, y_test = train_test_split(image_list, Y, test_size=0.01, random_state=111)
print(X_train[0])
print(y_train[0])
print(X_train[1])
print(y_train[1])


# モデルを生成してニューラルネットを構築
model = Sequential()
print("a")
model.add(Conv2D(32, (10, 10), input_shape=(X_train.shape[1:])))
model.add(Activation('relu'))
model.add(Conv2D(32, (10, 10)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
print("a")
model.add(Conv2D(64, (10, 10)))
model.add(Activation('relu'))
print("a")
model.add(Conv2D(64, (10, 10)))
model.add(Activation('relu'))
print("a")
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
print("a")
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))       # クラスは2個
# model.add(Activation('softmax'))
# モデルをコンパイル
model.compile(loss="mse", optimizer='adam')# metrics=["accuracy"])
print("compile")
plot_model(model, to_file="model10.png")

#visualize the filters
lays = model.layers
for i, l in enumerate(lays):
    print(i+1, l)
w1 = model.layers[0].get_weights()[0]
b1 = model.layers[0].get_weights()[1]
print(w1.shape, b1.shape)
print(model.layers[6].get_weights()[0].shape)

# 学習を実行。10%はテストに使用。
print(X_train.shape)
model.fit(X_train, y_train, epochs=3, batch_size=32, callbacks=[history])
# テスト用ディレクトリ(./data/train/)の画像でチェック。正解率を表示する。
total = 0.
ok_count = 0.

i=0
for i in range(len(X_test)):
    result = model.predict(np.array([X_test[i]]))
    print("label:", y_test[i], "result:", result[0][0])

    total += 1.

    # if label == result[0]:
    #     ok_count += 1.
    error.append(abs(float(y_test[i])-float(result[0][0])))

for i in range(len(campus_list)):
    result = model.predict(np.array([campus_list[i]]))
    print("name:", campus_name[i] , "result:", result[0][0])
    # if label == result[0]:
    #     ok_count += 1.
    prediction.append([campus_list[i], result[0][0]])


prediction.sort(key=itemgetter(2), reverse=True)
f=open('univ_ranking.csv', 'w')
writer = csv.writer(f)
for i in prediction:
    writer.writerow(i)
f.close()

#
# error_sum = 0
# for i in error:
# #     error_sum += i
# print("Average loss: ", abs(error_sum / total))

plt.plot(history.history['loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel("loss")
plt.legend('loss', loc='center right')
plt.savefig('10sec_history.png')



layer_num = 0; # 1st convolutional layer
print('Layer Name: {}'.format(model.layers[layer_num].get_config()['name']))
W = model.layers[layer_num].get_weights()[0]

W=W.transpose(3,2,0,1)
nb_filter, nb_channel, nb_row, nb_col = W.shape

#plot
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
plt.figure()
for i in range(nb_filter):
    im = W[i,0]
    # scaling images
    scaler = MinMaxScaler(feature_range=(0,255))
    im = scaler.fit_transform(im)

    plt.subplot(8,8,i+1)
    plt.axis('off')
    plt.imshow(im,cmap='gray')
plt.show()
plt.savefig('10sec_feature.png')
