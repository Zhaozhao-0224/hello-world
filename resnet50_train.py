import math
import os
import pandas as pd

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略烦人的警告
DATA_DIR = 'data_img'
TRAIN_DIR = os.path.join(DATA_DIR, 'data_train')
VALID_DIR = os.path.join(DATA_DIR, 'data_valid')
SIZE = (224, 224)
BATCH_SIZE = 16


def plot_training(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(len(acc))

  plt.plot(epochs, acc, 'r.')
  plt.plot(epochs, val_acc, 'r')
  plt.title('Training and validation accuracy')

  plt.figure()
  plt.plot(epochs, loss, 'r.')
  plt.plot(epochs, val_loss, 'r-')
  plt.title('Training and validation loss')

  plt.show()

if __name__ == "__main__":
    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

    gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
    val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)

    classes = list(iter(batches.class_indices))
    # 导入keras标准模型
    model_pretrained = keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False)
    # 创建模型:模型没有初始化权值
    base_model = keras.applications.resnet50.ResNet50(weights=None, include_top=False, pooling='max')
    # 定义网络的最后一层
    predictions = Dense(len(classes), activation="softmax")(base_model.layers[-1].output)
    # 定义最终模型
    finetuned_model = Model(inputs=base_model.input, outputs=predictions)
    # 导入权值：只导入前5层的权值
    for i in range(120):
        finetuned_model.layers[i].set_weights(model_pretrained.layers[i].get_weights())
    # 微调：指定前5层不训练
    for layer in finetuned_model.layers[:120]:
        layer.trainable = False
    finetuned_model.compile(optimizer=Adam(lr=0.001, decay=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    # print(len(finetuned_model.layers))
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes
    finetuned_model.summary()
    # # 当验证集上的loss不在减小（即减小的程度小于某个阈值）的时候停止继续训练。
    # early_stopping = EarlyStopping(patience=10)
    # checkpointer = ModelCheckpoint('resnet50_best.h5', verbose=1, save_best_only=True)
    # history = finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=20, callbacks=[early_stopping, checkpointer], validation_data=val_batches, validation_steps=num_valid_steps, class_weight='auto')
    # finetuned_model.save('resnet50_final.h5')
    #
    # # 图形化整个训练过程
    # plot_training(history)

