from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
import csv

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


# 学習用のデータを作る.
image_list = []
label_list = []

print(train_data[0][0])

# ./data/train 以下のorange,appleディレクトリ以下の画像を読み込む。
for dir in os.listdir("images/tr"):
    if dir == ".DS_Store":
        continue

    dir1 = "images/tr/" + dir

    # if dir == "apple":    # appleはラベル0
    #     label = 0
    # elif dir == "orange": # orangeはラベル1
    #     label = 1

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            for i in train_data:
                if i[0]==file:
                    label = i[2]
                    del i
                    break
            # 配列label_listに正解ラベルを追加(りんご:0 オレンジ:1)
            label_list.append(label)
            filepath = dir1 + "/" + file
            # 画像を25x25pixelに変換し、1要素が[R,G,B]3要素を含む配列の25x25の２次元配列として読み込む。
            # [R,G,B]はそれぞれが0-255の配列。
            image = np.random_integers((25,25,3))
            image_ = np.array(Image.open(filepath))
            image_ = np.array(Image.open(filepath).resize((25, 25)))
            if image_.shape[2]==4:
                for j in range(25):
                    for k in range(25):
                        image[j][k] = image_[j][k][0:3]
                # print(image.shape)
            # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
            image = image.transpose(2, 0, 1)
            # さらにフラットな1次元配列に変換。最初の1/3はRed、次がGreenの、最後がBlueの要素がフラットに並ぶ。
            image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
            # print(image.shape)
            # 出来上がった配列をimage_listに追加。
            image_list.append(image / 255.)

# # kerasに渡すためにnumpy配列に変換。
# print(len(image_list[0]))
image_list = np.array(image_list)
# for n in image_list:
    # print(n.shape)
# 
# ラベルの配列を1と0からなるラベル配列に変更
# 0 -> [1,0], 1 -> [0,1] という感じ。
# Y = to_categorical(label_list)
Y = label_list

# モデルを生成してニューラルネットを構築
model = Sequential()
model.add(Dense(200, input_dim=1875))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(1))
# model.add(Activation("softmax"))

# オプティマイザにAdamを使用
opt = Adam(lr=0.001)
# モデルをコンパイル
model.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])
# 学習を実行。10%はテストに使用。
print(image_list.shape)
model.fit(image_list, Y, epochs=10, batch_size=50, validation_split=0.1)

# テスト用ディレクトリ(./data/train/)の画像でチェック。正解率を表示する。
total = 0.
ok_count = 0.

for dir in os.listdir("images/te"):
    if dir == ".DS_Store":
        continue

    dir1 = "images/te/" + dir
    label = 0


    # for file in os.listdir(dir1):
    #     if file != ".DS_Store":
    #         for i in train_data:
    #             if i[0]==file:
    #                 label = i[1]
    #                 del i
    #                 break
    #         # 配列label_listに正解ラベルを追加(りんご:0 オレンジ:1)
    #         label_list.append(label)
    #         filepath = dir1 + "/" + file
    #         # 画像を25x25pixelに変換し、1要素が[R,G,B]3要素を含む配列の25x25の２次元配列として読み込む。
    #         # [R,G,B]はそれぞれが0-255の配列。
    #         image = np.random_integers((25,25,3))
    #         image_ = np.array(Image.open(filepath))
    #         image_ = np.array(Image.open(filepath).resize((25, 25)))
    #         if image_.shape[2]==4    #             for j in range(25):
    #                 for k in range(25):
    #                     image[j][k] = image_[j][k][0:3]
    #             print(image.shape)
    #         # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
    #         image = image.transpose(2, 0, 1)
    #         # さらにフラットな1次元配列に変換。最初の1/3はRed、次がGreenの、最後がBlueの要素がフラットに並ぶ。
    #         image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
    #         print(image.shape)
    #         # 出来上がった配列をimage_listに追加。
    #         image_list.append(image / 255.)

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            for i in test_data:
                if i[0]==file:
                    label = i[2]
                    del i
                    break
            label_list.append(label)
            filepath = dir1 + "/" + file
            image = np.random_integers((25,25,3))
            image_ = np.array(Image.open(filepath))
            image_ = np.array(Image.open(filepath).resize((25, 25)))
            if image_.shape[2]==4:
                for j in range(25):
                    for k in range(25):
                        image[j][k] = image_[j][k][0:3]
            # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
            image = image.transpose(2, 0, 1)
            image = image.reshape(1, image.shape[0] * image.shape[1] * image.shape[2]).astype("float32")[0]
            result = model.predict(np.array([image / 255.]))
            print("label:", label, "result:", result[0][0])

            total += 1.

            # if label == result[0]:
            #     ok_count += 1.
            error.append(float(label)-float(result[0][0]))

error_sum = 0
for i in error:
    error_sum += i

print("Average loss: ", abs(error_sum / total))
