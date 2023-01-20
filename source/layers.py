import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Reshape, Permute, Dropout, GlobalAveragePooling1D, Embedding
from tensorflow.keras.activations import softmax, linear
import tensorflow.keras.backend as K
import numpy as np
import logging

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

class scaledDotProductAttentionLayer(tf.keras.layers.Layer):
    def call(self, x, training):
        q, k, v = x
        qk = tf.matmul(q, k, transpose_b=True)/K.sqrt(tf.cast(K.shape(k)[-1], tf.float32))
        weights = softmax(qk, axis=-1)
        return tf.matmul(weights, v), weights

class multiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, head=12):
        super(multiHeadAttentionLayer, self).__init__()
        self.head = head
        self.permute = Permute((2, 1, 3))
        self.re1 = Reshape((-1, self.head, d_model//self.head))
        self.re2 = Reshape((-1, d_model))
        self.linear = Dense(d_model)
        
        self.attention = scaledDotProductAttentionLayer()               

    def call(self, x, training):
        q, k, v = x
        # subspace header
        q_s = self.permute(self.re1(q))
        k_s = self.permute(self.re1(k))
        v_s = self.permute(self.re1(v))
        
        # combine head
        head, weights = self.attention([q_s, k_s, v_s], training) # BTHC
        scaled_attention = self.permute(head) #BHTC
        concat_attention = self.re2(self.permute(scaled_attention)) #BTHC -> BTC
        multi_head = self.linear(concat_attention)
        return multi_head, weights

class mlpLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, output_dim):
        super(mlpLayer, self).__init__()
        self.d1 = Dense(hidden_dim, activation=gelu)#, kernel_regularizer=tf.keras.regularizers.l2(0.1))
        self.d2 = Dense(output_dim)#, kernel_regularizer=tf.keras.regularizers.l2(0.1))
    def call(self, x, training):
        x = self.d1(x)
        return self.d2(x)

class transformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, head_num=12, h_dim=3072):
        super(transformerBlock, self).__init__()
        self.q_d = Dense(d_model)
        self.k_d = Dense(d_model)
        self.v_d = Dense(d_model)
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.mlp = mlpLayer(h_dim, d_model)
        self.att = multiHeadAttentionLayer(d_model, head_num)
        
        self.drop = Dropout(0.1)
    
    def call(self, x, training):
        y = self.ln1(x)
        # multi head attention
        q = self.q_d(y) # query 
        k = self.k_d(y) # key
        v = self.v_d(y) # value
        y, weights = self.att([q, k, v], training)
#         y = self.drop(y)

        # skip connection
        x1 = x + y

        # MLP layer
        y = self.ln2(x1)
        y = self.mlp(y, training)
#         self.drop(y)
    
#         if training: # stocastic drop
#             stocastic_depth = tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32)
#             if stocastic_depth < 0.2:#stocastic_p
#                 return x, weights
#             else:
#                 return x1 + y, weights
                
        # skip connection
        return x1 + y, weights

class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = Dense(embed_dim)
        self.pos_embed = Embedding(input_dim=num_patch+1, output_dim=embed_dim)

    def call(self, patch, training):
        pos = tf.range(start=0, limit=self.num_patch+1, delta=1)
        return self.proj(patch, training) + self.pos_embed(pos, training)
    
class visionTransformerLayer(tf.keras.layers.Layer):
    def __init__(self, image_size, patch_size, d_model=768, layer_num=12, head_num=12, h_dim=3072, return_all_hidden=False):
        super(visionTransformerLayer, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.d_model = d_model
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_num = layer_num
        self.return_all_hidden = return_all_hidden
        
        # learnabel class embedding
#         self.class_emb = Dense(1, name='class_emb', activation='linear')
        self.cls = tf.Variable(
            name="cls_emb",
            initial_value=tf.zeros_initializer()(shape=(1, 1, self.d_model), dtype="float32"),
            trainable=True,
        )
        self.pos_emb = Embedding(input_dim=self.num_patches + 1, output_dim=d_model, name='pos_emb')
        
        self.patch_emb = [PatchEmbedding(self.num_patches, d_model, name=f'pat_emb_{i}') for i in range(layer_num)]
        
        self.per = Permute((2, 1))
#         self.class_emb = self.add_weight(shape=(1, 1, self.d_model),
#                                         initializer='random_normal',
#                                         trainable=True)
        
        # learnable position embedding
#         self.pos_emb = self.add_weight(shape=(1, 1, self.d_model), 
#                                       initializer='random_normal',
#                                       trainable=True)
        
        self.dense = Dense(d_model, activation='linear', name='image_emb')
        self.t_layer = [transformerBlock(d_model, head_num, h_dim) for i in range(layer_num)]
        self.layernorm = LayerNormalization()
        
    def call(self, x, training):
        # feature extraction
        # batch_size = tf.shape(x)[0]
       
        # resize image
        x = tf.image.resize(x, [self.image_size, self.image_size])
        
        # extract patch
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patches = Reshape((self.num_patches, -1))(patches)
        logging.debug(f'patches: {patches.shape}, {self.patch_size}, {self.num_patches}')
        x = self.dense(patches)
        logging.debug(f'patches: {x.shape}')
        
        pos = tf.range(start=0, limit=self.num_patches + 1, delta=1)
#         class_emb = self.per(self.class_emb(self.per(x)))
        pos_emb = self.pos_emb(pos)#self.per(self.pos_emb(self.per(x)))
#         class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.d_model])
        logging.debug(f'class_emb: {pos_emb.shape}')
        batch_size = tf.shape(x)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.d_model]),
            dtype=x.dtype,
        )
        x = tf.concat([cls_broadcasted, x], axis=1)
        x = x + pos_emb
        
        # transformer block    
        x_all = [] 
        weights_all = []
        
        for i in range(self.layer_num):
#             x = self.patch_emb[i](x_all[-1], training)
            x, w = self.t_layer[i](x, training)
            x_all.append(tf.identity(x))
            weights_all.append(tf.identity(w))
        
        x = self.layernorm(x)
        x_all[-1] = tf.identity(x)
        
        if self.return_all_hidden:
            return x_all, pos_emb, weights_all
        return x, pos_emb, weights_all

def visionTransformer(input_dim, output_dim, image_size=32, patch_size=8, d_model=768, layer_num=12, head_num=12, h_dim=3072, return_all_hidden=False):
    inputs = tf.keras.Input(shape=input_dim)
    ViT_layer = visionTransformerLayer(image_size, patch_size, d_model, layer_num, head_num, h_dim, return_all_hidden)
    feature, pos_emb, weights = ViT_layer(inputs)
    
    if return_all_hidden:
        y = GlobalAveragePooling1D()(feature[-1])
    else:
        y = GlobalAveragePooling1D()(feature)
    outputs = Dense(output_dim, activation='relu')(y)
    outputs = Dense(output_dim, activation='softmax')(outputs)
    logging.debug(f'output shape: {outputs.shape}')
    return tf.keras.Model(inputs, outputs, name='vit'), tf.keras.Model(inputs, [feature, pos_emb, weights], name='vit_layer')
    
    