import tensorflow as tf
import tensorflow.keras.layers as kl
import numpy as np
import copy
import logging
import math
# app_log = logging.getLogger('root')
logging.basicConfig(format='%(asctime)s|%(levelname)s|%(funcName)s(%(lineno)d)|%(message)s', level=logging.DEBUG)

class config():
    def __init__(self):
        self.d_model = 768
        self.beta = 2.
        self.top_k = 6
        self.teacher_start_decay = 0.9
        self.teacher_end_decay = 0.998

        self.start_epoch = 1
        self.end_epoch = 100
        
class emaOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, start_decay=0.998, end_decay=0.998, start_epoch=1, end_epoch=100, name="emaOptimizer", **kwargs):
        """Call super().__init__() and use _set_hyper() to store hyperparameters"""
        super().__init__(name, **kwargs)
        self._set_hyper("end_decay", end_decay) 
        self._set_hyper("learning_rate", start_decay)
        self._set_hyper("decay", start_decay)
        self._set_hyper("decay_delta", (end_decay - start_decay)/(end_epoch - start_epoch))
    
    def _create_slots(self, var_list):
        """For each model variable, create the optimizer variable associated with it.
        TensorFlow calls these optimizer variables "slots".
        For momentum optimization, we need one momentum slot per model variable.
        """
        for var in var_list:
            self.add_slot(var, "decay")
            
    @tf.function
    def _resource_apply_dense(self, grad, var):
        """Update the slots and perform one optimization step for one model variable
        """
        var_dtype = var.dtype.base_dtype
        end_decay_h = self._get_hyper("end_decay", var_dtype)
        decay_h = self._get_hyper("decay", var_dtype)
        decay_delta_h = self._get_hyper("decay_delta", var_dtype)
        lr_t = self._get_hyper("learning_rate", var_dtype)
        
        # Calculate exponetial rate
        local_step = tf.cast(self.iterations, var_dtype)
        lr_t = lr_t + decay_delta_h * local_step
        if tf.greater(lr_t, end_decay_h):
            lr_t = end_decay_h
        
        # Just copy weights of embedding layer
        if '_emb' in var.name:
            var.assign(grad)
        else:
            new_var_m = lr_t * var + (1 - lr_t) * grad
            var.assign(new_var_m)
        
    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "start_decay": self._serialize_hyperparameter("start_decay"),
            "end_decay": self._serialize_hyperparameter("end_decay"),
            "decay": self._serialize_hyperparameter("decay"),
            "start_epoch": self._serialize_hyperparameter("start_epoch"),
            "end_epoch": self._serialize_hyperparameter("end_epoch")
        }

def masking_image(image, mask_index, patch_size):
    patch_block = image.shape[0]//patch_size[0], image.shape[1]//patch_size[1]
    reshape_image = image.reshape(patch_block[0], patch_size[0], patch_block[1], patch_size[1], 3).swapaxes(1, 2).reshape(-1, patch_size[0], patch_size[1], 3)
    for i in mask_index:
        reshape_image[i] = np.zeros((patch_size[0], patch_size[1], 3))
    return reshape_image.reshape(patch_block[0], patch_block[1], patch_size[0], patch_size[1], 3).swapaxes(1, 2).reshape(image.shape)

def masking_block_index(image, patch_size, ratio):
    size = image.shape[:-1]
    patch_block_size = int(round(size[0]/patch_size[0])), int(round(size[0]/patch_size[0]))
    
    block_size = [0, 0]
    total_block_size = patch_block_size[0] * patch_block_size[1]
    min_patch_size_x = math.floor((total_block_size * ratio) / patch_block_size[1])
    block_size[0] = math.floor(np.random.randint(min_patch_size_x, patch_block_size[0] , 1))
    block_size[1] = math.floor((total_block_size * ratio) / block_size[0])
    if patch_block_size[0] == block_size[0]:
        x_start_index=0
    else:
        x_start_index = np.random.randint(patch_block_size[0] - block_size[0])
    if patch_block_size[1] == block_size[1]:
        y_start_index = 0
    else:
        y_start_index = np.random.randint(patch_block_size[1] - block_size[1])
    print(patch_block_size, block_size)
    selected_index = []
    for i in range(block_size[1]):
        for j in range(block_size[0]):
            selected_index.append(patch_block_size[1] * (y_start_index + i) + (x_start_index + j))
    return selected_index


def block_alg_2(patch_size_x, patch_size_y, rate=0.6, limit_size=2):
    total_block_size = patch_size_x * patch_size_y
    target_box_size = int(total_block_size * rate)
    mask_index = np.arange(total_block_size)
    np.random.shuffle(mask_index)
    maximum_size_x= int(patch_size_x * rate)
    maximum_size_y= int(patch_size_y * rate)
    # mask setting
    mask = np.zeros((patch_size_x, patch_size_y))
    while True:
        x_size = np.random.randint(limit_size, maximum_size_x, 1)[0]
        y_size = np.random.randint(limit_size, maximum_size_y, 1)[0]
        
        if np.sum(mask) + x_size * y_size > target_box_size:
            break

        # setting start points
        start_x, start_y = 0, 0
        if patch_size_x != x_size:
            start_x = int(np.random.uniform(low=0, high=patch_size_x - x_size))
        if patch_size_y != y_size:
            start_y = int(np.random.uniform(low=0, high=patch_size_y - y_size))
            
        mask[start_x:start_x + x_size, start_y:start_y + y_size] = 1
        
    reshape_mask = mask.reshape(patch_size_x*patch_size_y)
    mask_index = [i for i in range(len(reshape_mask)) if reshape_mask[i] == 1]    
    mask_index += [-1 for i in range(target_box_size - len(mask_index))]
    return mask, mask_index
    

def block_alg(patch_size_x, patch_size_y, rate=0.2):
    total_block_size = patch_size_x * patch_size_y
    target_box_size = int(total_block_size * rate)
    for i in range(target_box_size):
        if (target_box_size + i) % patch_size_x == 0:
            target_box_size += i
            break
        if (target_box_size - i) % patch_size_x == 0:
            target_box_size -= i
            break
    # if target_box_size % 2 == 1:
    #     target_box_size -= 1
    # make sample box sizes
    sample_boxes = []
    for i in range(1, patch_size_x + 1):
        tmp = int(target_box_size / i)
        if (tmp * i == target_box_size) & (tmp <= patch_size_y):
            sample_boxes.append([i, tmp])
    # select box size
    select_index = int(np.random.uniform(low=0, high=len(sample_boxes)))#tf.random.uniform([], minval=0, maxval=len(sample_boxes), dtype=tf.dtypes.int32)
    select_box = sample_boxes[select_index]
    
    # make start points    
    start_x, start_y = 0, 0
    if patch_size_x != select_box[0]:
        start_x = int(np.random.uniform(low=0, high=patch_size_x - select_box[0]))
    if patch_size_y != select_box[1]:
        start_y = int(np.random.uniform(low=0, high=patch_size_y - select_box[1]))
    
    # masking
    mask = np.zeros((patch_size_x, patch_size_y))
    mask[start_x:start_x + select_box[0], start_y:start_y + select_box[1]] += 1
    # mask = tf.scatter_nd(indexs, 1)
    # print(mask)
    # mask[start_x:start_x+select_box[0], start_y:start_y+select_box[1]] += 1
    reshape_mask = mask.reshape(patch_size_x*patch_size_y)
    mask_index = [i for i in range(len(reshape_mask)) if reshape_mask[i] == 1]
    return mask, mask_index

def image_aug(image, augmentation):
    import imgaug
    # Augmenters that are safe to apply to masks
    # Some, such as Affine, have settings that make them unsafe, so always
    # test your augmentation on masks
    MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                       "Fliplr", "Flipud", "CropAndPad",
                       "Affine", "PiecewiseAffine"]

    # Store shapes before augmentation to compare
    image_shape = image.shape
    # Make augmenters deterministic to apply similarly to images and masks
    det = augmentation.to_deterministic()
    image = det.augment_image(image)
    # Verify that shapes didn't change
    assert image.shape == image_shape, "Augmentation shouldn't change image size"
    return image

def generate_train(X_train, patch_size, patch_num, batch_size, mask_ratio = 0.5, aug=None):
    i = 0
    b = 0
    while True:
        if i > X_train.shape[0] - 1:
            i = 0
        tmp_indexs = np.arange(patch_num)
        np.random.shuffle(tmp_indexs)
        block_size = int(patch_num * mask_ratio)
        patch_size_x = X_train.shape[1] // patch_size[0]
        patch_size_y = X_train.shape[2] // patch_size[1]
        
        tmp_indexs = tmp_indexs[:block_size]
        _, tmp_indexs = block_alg_2(patch_size_x, patch_size_y, rate=mask_ratio)
        
        if b == 0:
            batch_X = np.zeros((batch_size, X_train.shape[1], X_train.shape[2], X_train.shape[3]), dtype='float32')
            batch_X_masked = np.zeros((batch_size, X_train.shape[1], X_train.shape[2], X_train.shape[3]), dtype='float32')
            batch_masked_index = np.zeros((batch_size, block_size), dtype='int32')#len(tmp_indexs)), dtype='int32')
            
        # normalization & augmentation
        if aug:
            image = image_aug(X_train[i], aug)
        else:
            image = X_train[i]
        batch_X[b] = image
        batch_X_masked[b] = masking_image(image, tmp_indexs, patch_size)#image_aug(X_train[i], aug)
        batch_masked_index[b] = np.array(tmp_indexs)
        
        b += 1
        i += 1

        if b >= batch_size:
            b = 0
            yield [batch_X, batch_X_masked], batch_masked_index

def generate_train_val(X_train, y_train, patch_size, patch_num, batch_size, mask_ratio = 0.5, aug=None):
    i = 0
    b = 0
    while True:
        if i > X_train.shape[0] - 1:
            i = 0
        tmp_indexs = np.arange(patch_num)
        np.random.shuffle(tmp_indexs)        
        tmp_indexs = tmp_indexs[:int(patch_num * mask_ratio)]
        _, tmp_indexs = block_alg_2(X_train.shape[1] // patch_size[0], X_train.shape[2]//patch_size[1], rate=mask_ratio)
        if b == 0:
            batch_X = np.zeros((batch_size, X_train.shape[1], X_train.shape[2], X_train.shape[3]), dtype='float32')
            batch_X_masked = np.zeros((batch_size, X_train.shape[1], X_train.shape[2], X_train.shape[3]), dtype='float32')
            batch_masked_index = np.zeros((batch_size, int(patch_num * mask_ratio)), dtype='int32')#len(tmp_indexs)), dtype='int32')
            batch_Y = np.zeros((batch_size,), dtype='int32')
            
        # normalization & augmentation
        if aug:
            image = image_aug(X_train[i], aug)
        else:
            image = X_train[i]
        batch_X[b] = image
        batch_X_masked[b] = masking_image(image, tmp_indexs, patch_size)
        batch_masked_index[b] = np.array(tmp_indexs)
        batch_Y[b] = y_train[i]
        
        b += 1
        i += 1

        if b >= batch_size:
            b = 0
            yield [batch_X, batch_X_masked], batch_masked_index, batch_Y

def smooth_l1_loss(y, predict, beta):
    mae = tf.math.abs(y - predict)
    return tf.math.reduce_mean(tf.keras.backend.switch(mae > beta, 
                                mae - (beta/2),
                                tf.math.pow(mae, 2)/(2 * beta)))

def e1(x):
    x, y = x
    return tf.einsum('ij,ij->i', x, y)

def e2(x):
    x, y = x
    return tf.einsum('ij,jk->ik', x, y)
    
class self_supervised_module(tf.keras.Model):
    def __init__(self, model, model2, cfg, optimizer=tf.keras.optimizers.Adam(1e-3)):
        super(self_supervised_module, self).__init__()
        self.student = model
        self.teacher = model2
        self.teacher.set_weights(copy.deepcopy(self.student.get_weights()))
        
        self.beta = cfg.beta
        self.k = cfg.top_k
        self.teacher_start_decay = cfg.teacher_start_decay
        self.teacher_end_decay = cfg.teacher_end_decay
        self.epoch = cfg.start_epoch
        self.end_epoch = cfg.end_epoch 
        self.decay_delta = (self.teacher_end_decay - self.teacher_start_decay)/(self.end_epoch + self.epoch)
        self.decay = self.teacher_start_decay
        
        self.optimizer = optimizer
        self.emaOptimizer = emaOptimizer(cfg.teacher_start_decay, cfg.teacher_end_decay, cfg.start_epoch, cfg.end_epoch)
        self.layerNorm_t = tf.keras.layers.LayerNormalization(axis=-1)
        self.layerNorm_s = tf.keras.layers.LayerNormalization(axis=-1)
        self.loss_f = tf.keras.losses.Huber(delta=self.beta)
        
        seq = [
            kl.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(cfg.d_model * 2, activation='relu', name='final_proj/dence1'),
#             tf.keras.layers.Dense(cfg.d_model * 2, activation='relu', name='final_proj/dence2'),
            tf.keras.layers.Dense(cfg.d_model, activation='linear', name='final_proj/dence2')
        ]
        seq2 = [
            kl.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(cfg.d_model, activation='linear', name='final_proj_t/dence')
        ]
        
        self.final_proj_s = tf.keras.Sequential(seq, name='final_proj')#(cfg.d_model, activation='relu', name='final_proj')        
        self.final_proj_t = tf.keras.Sequential(seq2, name='final_proj_t')
        
        self.lb1 = kl.Lambda(e1)
        self.lb2 = kl.Lambda(e2)
    
    def call(self, x):
        logging.debug(f'len(x): {len(x[1].shape)}')
        if len(x[1].shape) == 2:
            x, masked_index = x
        x_data, x_masked = x
#         logging.debug(f'x: {x_data}, x_m: {x_masked}')
        student_enc_mask = self.student(x_masked, training=False)
        student_enc = self.student(x_data, training=False)
        teacher_enc = self.teacher(x_data, training=False)
        return student_enc_mask, student_enc, teacher_enc

    def train_step(self, x):
        x, masked_index = x
        x_data, x_masked = x
        self.teacher.trainable = False
        
        with tf.GradientTape() as tape:
            student_enc, _, _ = self.student(x_masked, training=True) # masked input result
            teacher_enc, _, _ = self.teacher(x_data, training=False) # given by teacher top K layers
        
            # Normalize teacher_enc 
            teacher_enc = teacher_enc[-self.k:]
            
            # Nomalize each block
            teacher_enc = self.layerNorm_t(tf.keras.backend.permute_dimensions(tf.stack(teacher_enc), pattern=(1, 0, 2, 3))) # BLTC
            student_enc = self.layerNorm_s(student_enc)
            # logging.debug(f'teacher_enc shape m: {teacher_enc.shape}')
            
            # average top K block
            teacher_enc = tf.math.reduce_mean(teacher_enc, axis=1) # BTC
            
            logging.debug(f'stu : {student_enc}')
            logging.debug(f'tea : {teacher_enc}')
            
            logging.debug(f'after gather: {student_enc.shape}')
            
#             xs = tf.shape(student_enc)[-1]

            # smooth l1 norm
            loss = smooth_l1_loss(teacher_enc, student_enc, self.beta) #/ tf.math.sqrt(xs)
            logging.debug(f'loss: {loss}')
                   
        
        self.teacher.trainable = False
        trainable_vars = self.student.trainable_variables# + self.final_proj_s.trainable_variables + self.final_proj_t.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # teacher track student
        self.teacher.trainable = True
        modelWeights = self.student.trainable_weights
        trainable_vars_t = self.teacher.trainable_variables
        self.emaOptimizer.apply_gradients(zip(modelWeights, trainable_vars_t))

        return {'loss': loss}