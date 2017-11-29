#-*- coding:utf-8 -*-
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
from keras.optimizers import Adam, RMSprop, SGD
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
from keras import backend as K

image_size = 64
kernel = 4
dilation = 1
history = History()
nb_classes = 11

# with open('lenet.json', 'r') as file:
#     model_json = file.read()

# if os.path.getsize(history) > 0 :
# with open(history, 'rb') as f:
#     history = pickle.load(f)

f = open('images/metadata/oneHotData.csv', 'rt')
train_dataReader = csv.reader(f)
train_data = [ e for e in train_dataReader]
f.close()

g = open('/Users/Aoyama/Documents/B4/Memorability_data/database/hotMemorability.csv', 'rt')
test_dataReader = csv.reader(g)
our_data = [ e for e in test_dataReader]
g.close()

train_num = 0
test_num = 0


error=[]


# 学習用のデータを作る.
image_list = []
label_list = []
name_list = []

# ./data/train 以下のorange,appleディレクトリ以下の画像を読み込む。
for file in os.listdir("images/all"):
    if file != ".DS_Store":
        label=0
        for i in train_data:
            if i[0]==file:
                label = i[2]
                print('correct')
                del i
                break
        # 配列label_listに正解ラベルを追加(りんご:0 オレンジ:1)
        if label != 0:
            label_list.append(int(label))
            name_list.append(file)
            print(file)
            filepath = "images/all" + "/" + file
            # 画像を25x25pixelに変換し、1要素が[R,G,B]3要素を含む配列の25x25の２次元配列として読み込む。
            # [R,G,B]はそれぞれが0-255の配列。
            image = randint(255, size =(image_size,image_size,3))
            image_ = Image.open(filepath)
            if image_.mode != 'RGB':
                image_ = image_.convert('RGB')
            image_ = np.array(image_)
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

for file in os.listdir("images/correct"):
    if file != ".DS_Store":
        label=0
        print(file)
        for i in our_data:
            if i[0]==file:
                label = i[1]
                print('correct')
                del i
                break
        # 配列label_listに正解ラベルを追加(りんご:0 オレンジ:1)
        if label != 0:
            label_list.append(int(label))
            name_list.append(file)
            filepath = "images/correct" + "/" + file
            # 画像を25x25pixelに変換し、1要素が[R,G,B]3要素を含む配列の25x25の２次元配列として読み込む。
            # [R,G,B]はそれぞれが0-255の配列。
            image = randint(255, size =(image_size,image_size,3))
            image_ = Image.open(filepath)
            print(image_.mode)
            if image_.mode != 'RGB':
                image_ = image_.convert('RGB')
                print('not RGB')
            image_ = np.array(image_)
            # print(image_.shape)
            image_ = np.array(Image.open(filepath).resize((image_size, image_size), Image.BICUBIC))
            # print(file)
            # print(image_)
            # print(image_.shape)
            if len(image_.shape)==2:
                im = Image.open(filepath).resize((image_size, image_size))
                print('error')
                im.save('a.png')
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

# # kerasに渡すためにnumpy配列に変換。
# print(len(image_list[0]))
image_list = np.array(image_list)
image_list = image_list.astype("float32")
print(image_list[0])
image_list = image_list / 255 # for n in image_list:
    # print(n.shape)
#
# ラベルの配列を1と0からなるラベル配列に変更
# 0 -> [1,0], 1 -> [0,1] という感じ。
# Y = to_categorical(label_list)
Y = np.array(label_list)
Y = Y.astype("int")
print(Y)
#divide train and test data
X_train, X_test, Y_train, Y_test, z_train, z_test = train_test_split(image_list, Y, name_list, test_size=0.1, random_state=111)
y_train = np_utils.to_categorical(Y_train, nb_classes)
y_test = np_utils.to_categorical(Y_test, nb_classes)
y_train = y_train.astype('int')
y_test = y_test.astype('int')
print(X_train[0])
print(y_train[0])
print(X_train[1])
print(y_train[1])


# モデルを生成してニューラルネットを構築
model = Sequential()

model.add(Conv2D(32, (kernel, kernel), input_shape=(X_train.shape[1:]), dilation_rate=(dilation, dilation)))
first_layer = model.layers[-1]
input_img = first_layer.input
model.add(Activation('softmax'))
model.add(Conv2D(32, (kernel, kernel), dilation_rate=(dilation, dilation)))
model.add(Activation('softmax'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (kernel, kernel), dilation_rate=(dilation, dilation)))
model.add(Activation('softmax'))
model.add(Conv2D(64, (kernel, kernel), dilation_rate=(dilation, dilation)))
model.add(Activation('softmax'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
#
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('softmax'))
model.add(Dropout(0.5))
model.add(Dense(11))       # クラスは2個
model.add(Activation('softmax'))
# モデルをコンパイル
sgd = SGD(lr=0.1, decay=0, momentum=0.9, nesterov=True)
opt=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = Adam(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
plot_model(model, to_file="model10.png")

#visualize the filters
# lays = model.layers
# for i, l in enumerate(lays):
#     print(i+1, l)
# w1 = model.layers[0].get_weights()[0]
# b1 = model.layers[0].get_weights()[1]
# print(w1.shape, b1.shape)
# print(model.layers[6].get_weights()[0].shape)

# 学習を実行。10%はテストに使用。
print(image_list.shape)
model.fit(X_train, y_train, epochs=5, batch_size=16, callbacks=[history])
# テスト用ディレクトリ(./data/train/)の画像でチェック。正解率を表示する。
total = 0.
ok_count = 0.

i=0
for i in range(len(X_test)):
    result = model.predict_classes(np.array([X_test[i]]))
    def toNumber(a):
        ans = 0
        for i in range(len(a)):
            if a[i] == 1:
                ans = i
                break
        return ans
    label = toNumber(y_test[i])
    print("name:", z_test[i], "label:", label, "result:", result)
    print(result[0])
    total += 1.

    if label == result[0]:
        ok_count += 1.

score = model.evaluate(X_test, y_test)
print('score loss=', score[0])
print('score accuracy=',score[1])

# error_sum = 0
# for i in error:
#     error_sum += i

print("Accuracy: ", abs(ok_count / total))

plt.plot(history.history['loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel("loss")
plt.legend('loss', loc='center right')
plt.savefig('oneHot_10sec_history.png')

#活性化が最大となる画像
# layer_name = 'conv2d_3' # 可視化したい層
# filter_index = [] # 可視化したいフィルタ
# for i in range(16):
#     filter_index.append(i)
# layer_dict = dict([(layer.name, layer) for layer in model.layers])
# # 損失関数を作成
# for j in range(16):
#     layer_output = layer_dict[layer_name].output
#     loss = K.mean(layer_output[:, :, :, filter_index[j]])
#
#     # 勾配を計算。戻り値はリスト
#     grads = K.gradients(loss, input_img)[0]
#
#     # 勾配を規格化
#     grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon())
#
#     # input_imgを与えるとlossとgradsを返す関数を作成
#     iterate = K.function([input_img, K.learning_phase()], [loss, grads])
#
#     # ランダムに初期化
#     input_img_data = np.random.random((1, image_size, image_size, 3))
#     input_img_data = (input_img_data - 0.5) * 20 + 128
#     import scipy as sp
#     # sp.misc.imsave('filter/%s_random_%d.png' % (layer_name, filter_index[j]), input_img_data)
#     # gradient ascent
#     step=0.2
#     for i in range(10):
#         # print(input_img_data.shape)
#         # print(input_img_data[0][40][40])
#         loss_value, grads_value = iterate([input_img_data, 0])
#         # print(grads_value[0][40][40])
#         input_img_data += grads_value * step
#         # if abs(grads_value[0][40][40][0]) < 1e-6:
#             # break
#
#     def deprocess_image(x):
#         # 平均0, 標準偏差が0.1になるように規格化
#         x -= x.mean()
#         x /= (x.std() + 1e-5)
#         x *= 0.1
#
#         # 値が[0, 1]の範囲になるようにクリップ
#         x += 0.5
#         x = np.clip(x, 0, 1)
#
#         # RGBの配列に変換
#         x *= 255
#         # x = np.clip(x, 0, 255).astype('uint8')
#         return x
#
#     img = input_img_data[0]
#     img = deprocess_image(img)
#
#     sp.misc.imsave('oneHot_filter/%s_filter_%d.png' % (layer_name, filter_index[j]), img)
#
#
# layer_num = 0; # 1st convolutional layer
# print('Layer Name: {}'.format(model.layers[layer_num].get_config()['name']))
# W = model.layers[layer_num].get_weights()[0]
#
# W=W.transpose(3,2,0,1)
# nb_filter, nb_channel, nb_row, nb_col = W.shape
#
# #plot
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# plt.figure()
# for i in range(nb_filter):
#     im = W[i,0]
#     # scaling images
#     scaler = MinMaxScaler(feature_range=(0,255))
#     im = scaler.fit_transform(im)
#
#     plt.subplot(8,8,i+1)
#     plt.axis('off')
#     plt.imshow(im, vmin=0, vmax=255)#,cmap='gray')
# plt.show()
# plt.savefig('oneHot_10sec_feature.png')
