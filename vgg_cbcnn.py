import numpy as np
from keras import backend as K
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, concatenate, Lambda, Dense, Reshape,merge, concatenate
from tensorflow.keras import regularizers
from compact_bilinear_pooling import compact_bilinear_pooling_layer
 
def vgg_16_cbcnn(input_shape, no_classes, bilinear_output_dim, sum_pool=True, weight_decay_constant=5e-4,
                 multi_label=False, weights_path=None):

    weights_regularizer = regularizers.l2(weight_decay_constant)

    # Input layer
    img_input = Input(shape=input_shape, name='spectr_input')

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',
               kernel_regularizer=weights_regularizer)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',
               kernel_regularizer=weights_regularizer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',
               kernel_regularizer=weights_regularizer)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',
               kernel_regularizer=weights_regularizer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',
               kernel_regularizer=weights_regularizer)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2',
               kernel_regularizer=weights_regularizer)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',
               kernel_regularizer=weights_regularizer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',
               kernel_regularizer=weights_regularizer)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',
               kernel_regularizer=weights_regularizer)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',
               kernel_regularizer=weights_regularizer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',
               kernel_regularizer=weights_regularizer)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',
               kernel_regularizer=weights_regularizer)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',
               kernel_regularizer=weights_regularizer)(x)

    # Merge using compact bilinear method
    # dummy_tensor_for_output_dim = K.placeholder(shape=(bilinear_output_dim,))
    compact_bilinear_arg_list = [x, x]

    output_shape_x = x.get_shape().as_list()[1:]
    output_shape_cb = (output_shape_x[0], output_shape_x[1], bilinear_output_dim)    
    cbp = compact_bilinear_pooling_layer(x, x, bilinear_output_dim, sum_pool=True)

    # If sum_pool=True do a global sum pooling
    if sum_pool:
        # Since using tf. Hence 3rd would represent channels
        x = Lambda(lambda x: K.sum(x, axis=[1, 2]))(cbp)

    # Sign sqrt and L2 normalize result
    x = Lambda(lambda x: K.sign(x) * K.sqrt(K.abs(x)))(x)
    x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)

    # final dense layer
    if not multi_label:
        final_activation = 'softmax'
    else:
        final_activation = 'sigmoid'
    x = Dense(no_classes, activation=final_activation, name='softmax_layer', kernel_regularizer=weights_regularizer)(x)

    # Put together input and output to form model
    model = Model(inputs=[img_input], outputs=[x])
    if weights_path:
        model.load_weights(weights_path, by_name=True)
    return model


if __name__=='__main__':

    input_shape = (448, 448, 3,)
    no_classes = 128
    bilinear_output_dim = 8192

    vgg_weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model = vgg_16_cbcnn(input_shape, no_classes, bilinear_output_dim=bilinear_output_dim, sum_pool=True,
                         weights_path= False)

    print (model.summary())
