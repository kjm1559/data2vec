import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from numpy import dot
from numpy.linalg import norm
import numpy as np
import logging
import cv2
import io
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

def draw_loss(history, label, model, x_test, y_test):
    plt.clf()
    plt.cla()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Losses Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig(f'{label}.jpg')
    plt.clf()
    
    plt.plot(history.history["accuracy"], label="train_loss")
    plt.plot(history.history["val_accuracy"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test loss: {round(loss, 2)}")
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
#     print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    plt.title(f"Train and Validation Acc Over Epochs \w Eval acc :{round(accuracy * 100, 2)}", fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig(f'{label}_acc.jpg')
    plt.clf()


def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))


def draw_pos_emb(pos_emb, patch_shape, file_name):
    print(pos_emb.shape)
    plt.clf()
    plt.cla()
    pos = pos_emb[1:]
    pos_sim = []
    plt.title('transformer position embedding result')
    for i in range(pos.shape[0]):
        tmp_data = []
        for j in range(pos.shape[0]):
            tmp_data.append(cos_sim(pos[i], pos[j]))
        pos_sim.append(tmp_data)
#     print(pos_sim)
    pos_sim = np.array(pos_sim).reshape(pos.shape[0], patch_shape[0], patch_shape[1])
#     print(pos_sim)        
    for i in range(pos.shape[0]):
        plt.subplot(patch_shape[0], patch_shape[1], i + 1)
        plt.imshow(pos_sim[i])
        plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # plt.savefig(f'{file_name}.jpg')
    # plt.clf()
    return image
    
def draw_att(im, w, num, file_name):
    plt.clf()
    plt.cla()
    plt.figure(figsize=(8,6))
    for i in range(num):
        mask = weight_processing(w[:, i], im[i])
        plt.subplot(num, 3, 1 + i * 3)
        plt.imshow(im[i])
        plt.subplot(num, 3, 2 + i * 3)
        plt.imshow(mask.reshape(mask.shape[:-1]))
        plt.subplot(num, 3, 3 + i * 3)
        plt.imshow(mask * (im[i]))
    # plt.savefig(f'{file_name}.jpg')
    # plt.clf()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    return image
    
def weight_processing(w, image):
    w = np.array(w)
    logging.debug(f'weight shape : {w.shape}') # weight shape : (3, 4, 65, 65)
    
    grid_size = int(np.sqrt(w.shape[-1] - 1))
    
    reshaped = np.mean(w, axis=1)
    
    # From Section 3 in https://arxiv.org/pdf/2005.00928.pdf ...
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    reshaped = reshaped + np.eye(reshaped.shape[1])
    reshaped = reshaped / reshaped.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]

    # Recursively multiply the weight matrices
    v = reshaped[-1]
    for n in range(1, len(reshaped)):
        v = np.matmul(v, reshaped[-1 - n])

    # Attention from the output token to the input space.
    mask = v[0, 1:].reshape(grid_size, grid_size)
    mask = cv2.resize(mask / mask.max(), (image.shape[1], image.shape[0]))[
        ..., np.newaxis
    ]
    logging.debug(f'mask shape : {mask.shape}')
    return mask

def scatter_draw(data, labels, file_name, class_sample):
    plt.clf()
    plt.cla()
    data = data.reshape((data.shape[0], -1))
    pca = PCA(n_components=2)
    pc = pca.fit_transform(data)
    tsne = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(data)
    pc_y = np.c_[pc, tsne, np.array(labels)]
    df = pd.DataFrame(pc_y,columns=['PC1','PC2', 'TSNE1', 'TSNE2', 'diagnosis'])
    # df.to_csv('result.csv')
    fig, ax = plt.subplots(figsize=(16, 8))
    labels = df['diagnosis'].unique()
    # np.sort(labels)
    for i in range(len(labels)):
        # PCA
        ax = plt.subplot(1, 2, 1)
        plt.scatter(x=df[df['diagnosis'] == i]['PC1'], y=df[df['diagnosis'] == i]['PC2'], label=str(i))
        mean_x, mean_y = df[df['diagnosis'] == i][['PC1', 'PC2']].mean(axis=0)
        imagebox = OffsetImage(class_sample[int(i)], zoom=0.8)
        ab = AnnotationBbox(imagebox, (mean_x, mean_y))
        ax.add_artist(ab)
        
        # TSNE
        ax = plt.subplot(1, 2, 2)
        plt.scatter(x=df[df['diagnosis'] == i]['TSNE1'], y=df[df['diagnosis'] == i]['TSNE2'], label=str(i))
        mean_x, mean_y = df[df['diagnosis'] == i][['TSNE1', 'TSNE2']].mean(axis=0)
        imagebox = OffsetImage(class_sample[int(i)], zoom=0.8)
        ab = AnnotationBbox(imagebox, (mean_x, mean_y))
        ax.add_artist(ab)
        
    plt.legend()
    plt.title(file_name)
    plt.draw()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    return image
#     plt.savefig(f'{file_name}.jpg')
#     plt.clf()