from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, BatchNormalization
from keras.layers import Activation, add, multiply, Lambda
from keras.layers import AveragePooling2D, average, UpSampling2D, Dropout, AveragePooling3D
from keras.optimizers import Adam, SGD, RMSprop
from keras.initializers import glorot_normal, random_normal, random_uniform
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.losses import MSE
import nibabel as nb
import numpy as np


def UnetConv3D(input, outdim, is_batchnorm, name, kinit='glorot_normal'):
    x = Conv3D(outdim, (3, 3, 3), strides=(1, 1, 1), kernel_initializer=kinit, padding="same", name=name + '_1')(input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_act')(x)

    x = Conv3D(outdim, (3, 3, 3), strides=(1, 1, 1), kernel_initializer=kinit, padding="same", name=name + '_2')(x)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_act')(x)

    return x


def model_for(input_size, depth=2, base_filters=32):
    img_input = Input(shape=input_size, name='img_input')

    # first convolutions
    conv1 = UnetConv3D(img_input, base_filters, is_batchnorm=True, name='conv_input')
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    # create scales
    scales = []
    current_scale = img_input
    for i in range(1, depth):
        new_scale = AveragePooling3D(pool_size=(1, 2, 2), name='input_scale{}'.format(i))(current_scale)
        scales.append(new_scale)
        current_scale = new_scale

    current_pool = pool1
    for i in range(1, depth):
        n_filters = base_filters * (i + 1)
        print("filders =", n_filters)
        input_scaled = Conv3D(n_filters, (1, 3, 3), padding='same', activation='relu',
                              name="conv_scale{}".format(i))(scales[i - 1])
        concat_scaled = concatenate([input_scaled, current_pool], axis=4)
        conv = UnetConv3D(concat_scaled, n_filters, is_batchnorm=True, name='conv{}'.format(i))
        pool = MaxPooling3D(pool_size=(2, 2, 2))(conv)
        current_pool = pool

    n_filters_center = depth * base_filters
    print("Center filters", n_filters_center)
    center = UnetConv3D(current_pool, n_filters_center, is_batchnorm=True, name='center')

    model = Model(inputs=[img_input], outputs=scales)
    model.compile(optimizer=Adam(lr=0.1, decay=0.9), loss=MSE,
                  metrics=[MSE])
    return model

input_file = "/home/snarduzz/Data/97306792/Neuro_Cerebrale_64Ch/rcoreg_diff_trace_tra_TRACEW_97306792.nii"
file = nb.load(input_file).get_data()
_input = np.array([file, -file])
print(_input.shape)

model = model_for(_input.shape, depth=10)
output = model.predict(np.array([_input]))

"""
import matplotlib.pyplot as plt

layer = 8
for i in range(len(output)):
    scale = output[i]
    print(scale.shape)
    plt.imshow(scale[0,0][:,:,layer])
    plt.show()
"""