#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : yang chang jie 
# @Time    : 2020/10/24 13:46

import os
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU


class GAN(object):
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1

        # image shape
        self.image_shape = (self.img_rows, self.img_rows, self.channels)
        self.latent_dim = 100
        # adam 优化器
        optimizer = Adam(0.0002, 0.5)
        # 评估模型
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(

            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"]
        )

        self.generator = self.build_generator()

        gan_input = Input((self.latent_dim,))
        img = self.generator(gan_input)
        # 在训练generator的时候,不进行discriminator的训练
        self.discriminator.trainable = False
        # 对生成的假的图片进行预测
        validity = self.discriminator(img)
        self.combined = Model(gan_input, validity)
        # 生成模型  生成模型 要将评估模型的结果 作为输入
        self.combined.compile(

            loss="binary_crossentropy", optimizer=optimizer,
        )

    def build_generator(self):
        """生成网络 生成一串的数字"""
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        """
        BatchNormalization:批量标准化
        加速训练过程；
        可以使用较大的学习率；
        允许在深层网络中使用sigmoid这种易导致梯度消失的激活函数；
        具有轻微地正则化效果，以此可以降低dropout的使用。
        """

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.image_shape), activation="tanh"))
        model.add(Reshape(self.image_shape))
        """
        np.prod 将每个维度数进行相乘 得到一个标量
        >>> image_shape = (100, 100, 1)
        >>> np.prod(image_shape)
        10000
        """

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

    def build_discriminator(self):
        """评价网络 对输入的图片进行评价"""
        model = Sequential()
        model.add(Flatten(input_shape=self.image_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        # 判断真伪
        model.add(Dense(1, activation="sigmoid"))
        img = Input(shape=self.image_shape)
        validity = model(img)
        print(validity, "vailidity.....")
        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        # 获取数据
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # 进行标准化
        x_train = x_train / 127.5 - 1
        x_train = np.expand_dims(x_train, axis=3)

        # 创建标签
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(epochs):
            # 随机选取数据对discriminator进行训练
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_images = self.generator.predict(noise)

            # 损失函数
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # generator的训练
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            # 打印
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                # 保存生成模型生成的图片
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_images = self.generator.predict(noise)

        gen_images = 0.5 * gen_images + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_images[cnt, :, :, 0], cmap="gray")
                axs[i, j].axis("off")
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    if not os.path.exists("./images"):
        os.mkdir("./images")
    gan = GAN()
    gan.train(epochs=3000, batch_size=256, sample_interval=200)