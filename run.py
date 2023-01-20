import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(gpus)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)
from source.layers import visionTransformer
from source.self_supervised import config, self_supervised_module, generate_train, generate_train_val
from source.utils import draw_loss, draw_pos_emb, draw_att, scatter_draw
from imgaug import augmenters as iaa
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import logging

import os

class PerformancePlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, y_test, patch_size, batch_size):
        self.x_test = x_test[:1000]
        self.y_test = y_test[:1000]
        self.patch_num = (round(x_test.shape[1] / patch_size), round(x_test.shape[2] / patch_size))
        self.gen = generate_train_val(x_test, y_test, (patch_size, patch_size), self.patch_num[0] * self.patch_num[1], batch_size)
        self.class_samples = [x_test[y_test == i][0] for i in range(len(np.unique(y_test)))]
        
        self.tb = tf.summary.create_file_writer(f'./logs/pretrain/')
        
    def on_epoch_end(self, epoch, logs={}):
        sm_ = []
        s_ = []
        t_ = []
        y_ = []
        for x in self.gen:
            x, m, y = x
            sm, s, t = self.model([x, m])
            if len(sm_) == 0:
                sm_, s_, t_ = sm[0].numpy(), s[0].numpy(), t[0][-1].numpy()
                y_ = y
                img_att_s = draw_att(x[0], tf.stack(s[-1]), 5, f'./image_results/att_result_{epoch+1}')
                img_att_sm = draw_att(x[0], tf.stack(sm[-1]), 5, f'./image_results/mask_att_result_{epoch+1}')
                img_att_t = draw_att(x[0], tf.stack(t[-1]), 5, f'./image_results/teacher_att_result_{epoch+1}')
                logging.debug(f'pos_emb shape : {s[1].shape}')
                img_pos_emb = draw_pos_emb(s[1], self.patch_num, f'./image_results/pos_emb_result_{epoch+1}')
            else:
                sm_ = np.concatenate([sm_, sm[0].numpy()], axis=0)
                s_ = np.concatenate([s_, s[0].numpy()], axis=0)
                t_ = np.concatenate([t_, t[0][-1].numpy()], axis=0)
                y_ = np.concatenate([y_, y], axis=0)
            if len(sm_) > len(self.y_test):
                break
        
        img_sm = scatter_draw(sm_, y_, f'masked_student_{epoch}', self.class_samples)
        img_s = scatter_draw(s_, y_, f'student_{epoch}', self.class_samples)
        img_t = scatter_draw(t_, y_, f'teacher_{epoch}', self.class_samples)
#         plt.savefig(f'./image_results/result_{epoch+1}.jpg')
#         plt.clf()
        
        with self.tb.as_default():
            tf.summary.image("attentions", [img_att_sm, img_att_s, img_att_t], max_outputs=3, step=epoch)
            tf.summary.image("pos_emb", [img_pos_emb], max_outputs=1, step=epoch)
            tf.summary.image("result", [img_sm, img_s, img_t], max_outputs=3, step=epoch)
            # self.tb.add_eumbedding(
            for key in logs.keys():
                tf.summary.scalar(key, logs[key], step=epoch)


if __name__ == '__main__':
    # self-supervise learning by cifar100
    if sys.argv[1] == 'mnist':
        reduce_cifar100 = tf.keras.datasets.mnist
    elif sys.argv[1] == 'cifar10':
        reduce_cifar100 = tf.keras.datasets.cifar10
    elif sys.argv[1] == 'cifar100':
        reduce_cifar100 = tf.keras.datasets.cifar100
    (X_train, y_train), (X_test, y_test) = reduce_cifar100.load_data()
    
    print(X_train.shape)
    
    if len(X_train.shape) != 4:
        X_train = np.stack([X_train] * 3, axis=-1)
        X_test = np.stack([X_test] * 3, axis=-1)
    
    if sys.argv[1] == 'cifar100':
        y_train = np.squeeze(np.eye(100)[y_train])
        y_test = np.squeeze(np.eye(100)[y_test])
    else:
        y_train = np.squeeze(np.eye(10)[y_train])
        y_test = np.squeeze(np.eye(10)[y_test])
    
    print(y_train.shape, X_train.shape, y_train.shape)

    #normalization
    X_train = X_train/255.
    X_test = X_test/255.
        
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    cd = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=1e-3, first_decay_steps=4, t_mul=2.0, m_mul=0.9, alpha=1e-3)
    ls = tf.keras.callbacks.LearningRateScheduler(cd)
    if sys.argv[1] == 'mnist':
        image_size=28
    else:
        image_size=32
    patch_size=4
    
    # transformer options
    d_model = 128
    layer_num = 3
    head_num = 4
    h_dim = 512
    
#     d_model=768, layer_num=12, head_num=12, h_dim=3072

    if not(os.path.isdir('./image_results')):
        os.mkdir('./image_results')
    
    if 'pretrain' in sys.argv[2]:
        # model is base/16
        _, header = visionTransformer(X_train.shape[1:], y_train.shape[-1], image_size=image_size, patch_size=patch_size, return_all_hidden=False, 
                                      layer_num=layer_num, d_model=d_model, head_num=head_num, h_dim=h_dim)
        _, header2 = visionTransformer(X_train.shape[1:], y_train.shape[-1], image_size=image_size, patch_size=patch_size, return_all_hidden=True, 
                                       layer_num=layer_num, d_model=d_model, head_num=head_num, h_dim=h_dim)

        cfg = config()
        cfg.epoch = 50
        cfg.batch_size = 2048#256
        cfg.end_epoch = int(X_train.shape[0]/cfg.batch_size + 0.5) * cfg.epoch
        cfg.teacher_start_decay = 0.9998
        cfg.teacher_end_decay = 0.9998
        cfg.k = 2
        cfg.d_model = d_model
        cfg.beta = 2
        cfg.mask_ratio = .6

        ss_model = self_supervised_module(header, header2, cfg)
        ss_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(1e-3))#, callbacks=[ls])
    #     print('header 1 :', ss_model.student.get_weights()[-2])
    #     print('header 2 :', ss_model.teacher.get_weights()[-2])

        # for test
    #     ss_model.train_step(([X_train[:2], X_train[:2]], np.array([[0, 1], [2, 3]])))
    #     exit -1

        augmentation = iaa.SomeOf((0, 2), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            # iaa.OneOf([iaa.Affine(rotate=90),
            #            iaa.Affine(rotate=180),
            #            iaa.Affine(rotate=270)]),
#             iaa.Multiply((0.8, 1.5)),
            iaa.GaussianBlur(sigma=(0.0, 5.0))
        ])

        gen = generate_train(X_train, (patch_size, patch_size), X_train.shape[1]//patch_size * X_train.shape[2]//patch_size, cfg.batch_size, aug=augmentation, mask_ratio=cfg.mask_ratio)
        
        # val_samples
        labels = np.unique(y_test.argmax(axis=1))
        print(labels)
        val_data = []
        val_y = []
        for k in labels:
            if len(val_data) == 0:
                print(y_test.argmax(axis=1))
                val_data = X_test[y_test.argmax(axis=1) == k][:200]
            else:
                val_data = np.concatenate([val_data, X_test[y_test.argmax(axis=1) == k][:200]], axis=0)
            val_y += [k] * min(X_test[y_test.argmax(axis=1) == k][:200].shape[0], 200)
        print(len(val_y), X_test.shape)
        index = np.arange(len(val_y), dtype='int32')
        np.random.shuffle(index)
        val_data = val_data[index]
        val_y = np.array(val_y)[index]
            
        ppc = PerformancePlotCallback(val_data, val_y, patch_size, 128)


        history = ss_model.fit(gen, steps_per_epoch=int(X_train.shape[0]/cfg.batch_size + 0.5), batch_size=cfg.batch_size, epochs=cfg.epoch, callbacks=[ppc, ls])
        ss_model.student.save_weights(f'{sys.argv[1]}_self_supervise.h5')
    
    # fine tune
    model, header = visionTransformer(X_train.shape[1:], y_train.shape[-1], image_size=image_size, patch_size=patch_size,
                                      layer_num=layer_num, d_model=d_model, head_num=head_num, h_dim=h_dim)
    
    if 'finetune' in sys.argv[2]:
        print(header.get_weights()[-2])
        header.load_weights(f'{sys.argv[1]}_self_supervise.h5', True)
        header.trainable = False
        print(header.get_weights()[-2])
    
    inputs = tf.keras.Input(shape=X_train.shape[1:])
    
    feature, pos_emb, weights = header(inputs)
    
    
    y = tf.keras.layers.GlobalAveragePooling1D()(feature)
    outputs = tf.keras.layers.Dense(y_train.shape[-1], activation='softmax')(y)
    model = tf.keras.Model(inputs, outputs, name='vit')
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['accuracy'])
    
    if 'finetune' in sys.argv[2]: 
        header.trainable = False
        model.summary()
        tb = tf.keras.callbacks.TensorBoard(log_dir='logs/finetune')
        history = model.fit(X_train, y_train, validation_split=0.2, batch_size=64, epochs=20, callbacks=[es, tb])

        model.evaluate(X_test, y_test)
        draw_loss(history, sys.argv[1] + '_head_' + sys.argv[2], model, X_test, y_test)
        
        header.trainable = True
    
    # fine tune
    cd = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=1e-3, first_decay_steps=5, t_mul=2.0, m_mul=0.9, alpha=1e-6)
    ls = tf.keras.callbacks.LearningRateScheduler(cd)
    header.trainable = True
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])
    model.summary()
    tb = tf.keras.callbacks.TensorBoard(log_dir=f'logs/train_{sys.argv[2]}')
    history = model.fit(X_train, y_train, validation_split=0.2, batch_size=64, epochs=20, callbacks=[es, tb])
    
    num = 5
    _, pos, w = header(X_test[:num], training=False)
    w = np.array(w)
    print(w.shape, num)
    draw_att(X_test[:num], w, num, f'./image_results/{sys.argv[1]}_{sys.argv[2]}_final_att_result')
    draw_pos_emb(pos, (round(X_train.shape[1]/patch_size), round(X_train.shape[2]/patch_size)), f'./image_results/{sys.argv[1]}_{sys.argv[2]}_pos_emb_result')
       

    draw_loss(history, f'{sys.argv[1]}_{sys.argv[2]}_', model, X_test, y_test)