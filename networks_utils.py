# Import stuff
import sys
import keras
import tensorflow as tf

# Additional imports from keras
import keras.backend as K
from keras                      import regularizers
from keras                      import optimizers
from keras.models               import Model
from keras.layers               import Input
from keras.layers               import concatenate
from keras.layers               import Conv2D
from keras.layers               import MaxPooling2D
from keras.layers               import AveragePooling2D
from keras.layers               import Flatten
from keras.layers               import Lambda
from keras.layers               import Dense
from keras.layers               import Activation
from keras.layers               import Conv2DTranspose
from keras.layers.convolutional import ZeroPadding2D
from keras.callbacks            import EarlyStopping
from keras.callbacks            import ModelCheckpoint
from tensorflow.keras.losses    import MeanSquaredError, BinaryCrossentropy

# Custom imports
from datasets_utils import *
from params import *



### I/O convolutional layer
def io_conv_2D(x,
               seed,
               filters     = 8,
               kernel_size = 3,
               strides     = 1,
               padding     = 'same',
               activation  = 'relu'):

    sigma = np.sqrt(2. / ((kernel_size**2) * K.int_shape(x)[-1]))

    x = Conv2D(filters     = filters,
               kernel_size = kernel_size,
               kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=sigma, seed=seed),
               strides     = strides,
               padding     = padding,
               activation  = activation)(x)

    return x

### ************************************************
### I/O max-pooling layer
def io_maxp_2D(x,
               pool_size = 2,
               strides   = 2):

    x = MaxPooling2D(pool_size = pool_size,
                     strides   = strides)(x)

    return x

### ************************************************
### I/O convolutional transposed layer
def io_conv_2D_transp(in_layer,
                      n_filters,
                      kernel_size,
                      stride_size,
                      seed):

    sigma = np.sqrt(2. / ((kernel_size**2) * K.int_shape(in_layer)[-1]))
    out_layer = Conv2DTranspose(filters=n_filters,
                                kernel_size=kernel_size,
                                kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=sigma, seed=seed),
                                #keras.initializers.RandomNormal(mean=0.0, stddev=sigma, seed=1),
                                strides=stride_size,
                                padding='same')(in_layer)

    return out_layer


### Return a constant tensor with the same shape as input
def constant_layer(in_layer):
    return tf.ones_like(in_layer)

### ************************************************
### I/O concatenate + zero-pad
def io_concat_pad(in_layer_1,
                  in_layer_2,
                  axis):

    # Compute padding sizes
    shape1_x  = np.asarray(keras.backend.int_shape(in_layer_1)[1])
    shape1_y  = np.asarray(keras.backend.int_shape(in_layer_1)[2])
    shape2_x  = np.asarray(keras.backend.int_shape(in_layer_2)[1])
    shape2_y  = np.asarray(keras.backend.int_shape(in_layer_2)[2])
    dx        = shape2_x - shape1_x
    dy        = shape2_y - shape1_y

    # Pad and concat
    pad_layer = ZeroPadding2D(((dx,0),(dy,0)))(in_layer_1)
    out_layer = concatenate([pad_layer, in_layer_2], axis=axis)

    return out_layer

### I/O zero-pad
def io_pad(in_layer_1,
           in_layer_2):

    # Compute padding sizes
    shape1_x  = np.asarray(keras.backend.int_shape(in_layer_1)[1])
    shape1_y  = np.asarray(keras.backend.int_shape(in_layer_1)[2])
    shape2_x  = np.asarray(keras.backend.int_shape(in_layer_2)[1])
    shape2_y  = np.asarray(keras.backend.int_shape(in_layer_2)[2])
    dx        = shape2_x - shape1_x
    dy        = shape2_y - shape1_y

    # Pad and concat
    pad_layer = ZeroPadding2D(((dx,0),(dy,0)))(in_layer_1)
    
    return pad_layer

	
### A custom loss function: the weighted sum of reconstruction MSE and the flow prediciton MSE
def custom_loss(i):
    
    def my_loss(y_true, y_pred):
        
        # entropy loss on the shape channel
        shape_true = y_true[:,:,:,0]
        shape_pred = y_pred[:,:,:,0]
        #loss_shape = binary_focal_loss_fixed(shape_true, shape_pred)
        loss_shape = MeanSquaredError()
        loss_shape = loss_shape(shape_true, shape_pred)

        # mse on the flow channels
        flow_true  = y_true[:,:,:,1:]
        flow_pred  = y_pred[:,:,:,1:]
        #loss_flow  = K.mean(K.sum(K.square(flow_true - flow_pred), axis=(1,2,3)))
        loss_flow  = MeanSquaredError()
        loss_flow  = loss_flow(flow_true, flow_pred)
        
        loss       = tf.multiply(loss_shape, i) + loss_flow

        return loss

    return my_loss

keras.losses.custom_loss = custom_loss




### Dual-AE model
def Dual_AE(train_im,
        train_sol,
        valid_im,
        valid_sol,
        n_filters_initial,
        kernel_size,
        kernel_transpose_size,
        pool_size,
        stride_size,
        learning_rate,
        batch_size,
        n_epochs,
        height,
        width,
        n_channels,
        weight,
        seed):

    # Generate inputs
    conv0 = Input((height,width,n_channels))

    ##################### Encoder begins ##################
    # 2 convolutions + maxPool
    conv1 = io_conv_2D(conv0, seed, n_filters_initial*(2**0), kernel_size)
    conv1 = io_conv_2D(conv1, seed, n_filters_initial*(2**0), kernel_size)
    pool1 = io_maxp_2D(conv1, pool_size)
    # 2 convolutions + maxPool
    conv2 = io_conv_2D(pool1, seed, n_filters_initial*(2**1), kernel_size)
    conv2 = io_conv_2D(conv2, seed, n_filters_initial*(2**1), kernel_size)
    pool2 = io_maxp_2D(conv2, pool_size)
    # 2 convolutions + maxPool
    conv3 = io_conv_2D(pool2, seed, n_filters_initial*(2**2), kernel_size)
    conv3 = io_conv_2D(conv3, seed, n_filters_initial*(2**2), kernel_size)
    pool3 = io_maxp_2D(conv3, pool_size)
    # 2 convolutions + maxPool
    conv4 = io_conv_2D(pool3, seed, n_filters_initial*(2**3), kernel_size)
    conv4 = io_conv_2D(conv4, seed, n_filters_initial*(2**3), kernel_size)
    pool4 = io_maxp_2D(conv4, pool_size)
    # 2 convolutions
    conv5 = io_conv_2D(pool4, seed, n_filters_initial*(2**4), kernel_size)
    conv5 = io_conv_2D(conv5, seed, n_filters_initial*(2**4), kernel_size)
    ##################### Encoder ends ##################

    ##################### Decoder shape begins ##################
    # 1 transpose convolution + 2 convolutions
    pre6  = io_conv_2D_transp(conv5, n_filters_initial*(2**3), kernel_transpose_size, stride_size, seed=seed)
    up6   = io_pad(pre6, conv4)
    conv6 = io_conv_2D(up6, seed, n_filters_initial*(2**3), kernel_size)
    conv6 = io_conv_2D(conv6, seed, n_filters_initial*(2**3), kernel_size)
    # 1 transpose convolution + 2 convolutions
    pre7  = io_conv_2D_transp(conv6 , n_filters_initial*(2**2), kernel_transpose_size, stride_size, seed=seed)
    up7   = io_pad(pre7, conv3)
    conv7 = io_conv_2D(up7, seed, n_filters_initial*(2**2), kernel_size)
    conv7 = io_conv_2D(conv7, seed, n_filters_initial*(2**2), kernel_size)
    # 1 transpose convolution + 2 convolutions
    pre8  = io_conv_2D_transp(conv7, n_filters_initial*(2**1), kernel_transpose_size, stride_size, seed=seed)
    up8   = io_pad(pre8, conv2)
    conv8 = io_conv_2D(up8, seed, n_filters_initial*(2**1), kernel_size)
    conv8 = io_conv_2D(conv8, seed, n_filters_initial*(2**1), kernel_size)
    # 1 transpose convolution + 2 convolutions
    pre9  = io_conv_2D_transp(conv8, n_filters_initial*(2**0), kernel_transpose_size, stride_size, seed=seed)
    up9   = io_pad(pre9, conv1)
    conv9 = io_conv_2D(up9, seed, n_filters_initial*(2**0), kernel_size)
    conv9 = io_conv_2D(conv9, seed, n_filters_initial*(2**0), kernel_size)
    ##################### Decoder shape ends #####################


    ##################### Decoder flow begins ##################
    # 1 transpose convolution and 2 convolutions
    pre16  = io_conv_2D_transp(conv5, n_filters_initial*(2**3), kernel_transpose_size, stride_size, seed=seed)
    cnst_4 = Lambda(constant_layer)(conv4)
    up16   = io_concat_pad(pre16, cnst_4, 3)
    conv16 = io_conv_2D(up16, seed, n_filters_initial*(2**3), kernel_size)
    conv16 = io_conv_2D(conv16, seed, n_filters_initial*(2**3), kernel_size)
    # 1 transpose convolution and 2 convolutions
    pre17  = io_conv_2D_transp(conv16, n_filters_initial*(2**2), kernel_transpose_size, stride_size, seed=seed)
    cnst_3 = Lambda(constant_layer)(conv3)
    up17   = io_concat_pad(pre17,cnst_3, 3)
    conv17 = io_conv_2D(up17, seed, n_filters_initial*(2**2), kernel_size)
    conv17 = io_conv_2D(conv17, seed, n_filters_initial*(2**2), kernel_size)
    # 1 transpose convolution and 2 convolutions
    pre18  = io_conv_2D_transp(conv17, n_filters_initial*(2**1), kernel_transpose_size, stride_size, seed=seed)
    cnst_2 = Lambda(constant_layer)(conv2)
    up18   = io_concat_pad(pre18, cnst_2, 3)
    conv18 = io_conv_2D(up18, seed, n_filters_initial*(2**1), kernel_size)
    conv18 = io_conv_2D(conv18, seed, n_filters_initial*(2**1), kernel_size)
    # 1 transpose convolution and 2 convolutions
    pre19  = io_conv_2D_transp(conv18, n_filters_initial*(2**0), kernel_transpose_size, stride_size, seed=seed)
    cnst_1 = Lambda(constant_layer)(conv1)
    up19   = io_concat_pad(pre19, cnst_1, 3)
    conv19 = io_conv_2D(up19, seed, n_filters_initial*(2**0), kernel_size)
    conv19 = io_conv_2D(conv19, seed, n_filters_initial*(2**0), kernel_size)
    ##################### Decoder flow ends #####################


    # final 1x1 convolution
    conv10 = io_conv_2D(conv9, seed, 1, 1)
    conv20 = io_conv_2D(conv19, seed, 3, 1)
    Output = io_concat_pad(conv10, conv20, 3)
    # construct model
    model = Model(inputs=[conv0], outputs=[Output])



    # Print info about model
    model.summary()

    # Set training parameters
    model.compile(loss=custom_loss(weight), optimizer=optimizers.Adam(lr=learning_rate, decay=0.0002))

    early = EarlyStopping(monitor  = 'val_loss',
                          mode     = 'min',
                          verbose  = 0,
                          patience = 10)
    check = ModelCheckpoint('Depth5/{}_{}_{}/best_{}_{}_{}.h5'.format(weight, n_filters_initial, seed, weight, n_filters_initial, seed),
                            monitor           = 'val_loss',
                            mode              = 'min',
                            verbose           = 0,
                            save_best_only    = True,
                            save_weights_only = False)

    # Train network
    with tf.device('/gpu:0'):
        train_model = model.fit(train_im, train_sol,
                                batch_size=batch_size, epochs=n_epochs,
                                validation_data=(valid_im, valid_sol),
                                callbacks       = [early, check])

    return(model, train_model)


## U_Dual_AE holds skip connections from encoder to flow decoder
def U_Dual_AE(train_im,
        train_sol,
        valid_im,
        valid_sol,
        n_filters_initial,
        kernel_size,
        kernel_transpose_size,
        pool_size,
        stride_size,
        learning_rate,
        batch_size,
        n_epochs,
        height,
        width,
        n_channels,
        weight,
        seed):

    # Generate inputs
    conv0 = Input((height,width,n_channels))
    
    
    ##################### Encoder begins ##################
    # 2 convolutions + maxPool
    conv1 = io_conv_2D(conv0, seed, n_filters_initial*(2**0), kernel_size)
    conv1 = io_conv_2D(conv1, seed, n_filters_initial*(2**0), kernel_size)
    pool1 = io_maxp_2D(conv1, pool_size)
    # 2 convolutions + maxPool
    conv2 = io_conv_2D(pool1, seed, n_filters_initial*(2**1), kernel_size)
    conv2 = io_conv_2D(conv2, seed, n_filters_initial*(2**1), kernel_size)
    pool2 = io_maxp_2D(conv2, pool_size)
    # 2 convolutions + maxPool
    conv3 = io_conv_2D(pool2, seed, n_filters_initial*(2**2), kernel_size)
    conv3 = io_conv_2D(conv3, seed, n_filters_initial*(2**2), kernel_size)
    pool3 = io_maxp_2D(conv3, pool_size)
    # 2 convolutions + maxPool
    conv4 = io_conv_2D(pool3, seed, n_filters_initial*(2**3), kernel_size)
    conv4 = io_conv_2D(conv4, seed, n_filters_initial*(2**3), kernel_size)
    pool4 = io_maxp_2D(conv4, pool_size)
    # 2 convolutions
    conv5 = io_conv_2D(pool4, seed, n_filters_initial*(2**4), kernel_size)
    conv5 = io_conv_2D(conv5, seed, n_filters_initial*(2**4), kernel_size)
    ##################### Encoder ends ##################


    ##################### Decoder shape begins ##################
    # 1 transpose convolution + 2 convolutions
    pre6  = io_conv_2D_transp(conv5, n_filters_initial*(2**3), kernel_transpose_size, stride_size, seed=seed)
    up6   = io_pad(pre6, conv4)
    conv6 = io_conv_2D(up6, seed, n_filters_initial*(2**3), kernel_size)
    conv6 = io_conv_2D(conv6, seed, n_filters_initial*(2**3), kernel_size)
    # 1 transpose convolution + 2 convolutions
    pre7  = io_conv_2D_transp(conv6 , n_filters_initial*(2**2), kernel_transpose_size, stride_size, seed=seed)
    up7   = io_pad(pre7, conv3)
    conv7 = io_conv_2D(up7, seed, n_filters_initial*(2**2), kernel_size)
    conv7 = io_conv_2D(conv7, seed, n_filters_initial*(2**2), kernel_size)
    # 1 transpose convolution + 2 convolutions
    pre8  = io_conv_2D_transp(conv7, n_filters_initial*(2**1), kernel_transpose_size, stride_size, seed=seed)
    up8   = io_pad(pre8, conv2)
    conv8 = io_conv_2D(up8, seed, n_filters_initial*(2**1), kernel_size)
    conv8 = io_conv_2D(conv8, seed, n_filters_initial*(2**1), kernel_size)
    # 1 transpose convolution + 2 convolutions
    pre9  = io_conv_2D_transp(conv8, n_filters_initial*(2**0), kernel_transpose_size, stride_size, seed=seed)
    up9   = io_pad(pre9, conv1)
    conv9 = io_conv_2D(up9, seed, n_filters_initial*(2**0), kernel_size)
    conv9 = io_conv_2D(conv9, seed, n_filters_initial*(2**0), kernel_size)
    ##################### Decoder shape ends #####################

    ##################### Decoder flow begins ##################
    # 1 transpose convolution and concat + 2 convolutions
    pre16  = io_conv_2D_transp(conv5, n_filters_initial*(2**3), kernel_transpose_size, stride_size, seed=seed)
    up16   = io_concat_pad(pre16, conv4, 3)
    conv16 = io_conv_2D(up16, seed, n_filters_initial*(2**3), kernel_size)
    conv16 = io_conv_2D(conv16, seed, n_filters_initial*(2**3), kernel_size)
    # 1 transpose convolution and concat + 2 convolutions
    pre17  = io_conv_2D_transp(conv16, n_filters_initial*(2**2), kernel_transpose_size, stride_size, seed=seed)
    up17   = io_concat_pad(pre17, conv3, 3)
    conv17 = io_conv_2D(up17, seed, n_filters_initial*(2**2), kernel_size)
    conv17 = io_conv_2D(conv17, seed, n_filters_initial*(2**2), kernel_size)
    # 1 transpose convolution and concat + 2 convolutions
    pre18  = io_conv_2D_transp(conv17, n_filters_initial*(2**1), kernel_transpose_size, stride_size, seed=seed)
    up18   = io_concat_pad(pre18, conv2, 3)
    conv18 = io_conv_2D(up18, seed, n_filters_initial*(2**1), kernel_size)
    conv18 = io_conv_2D(conv18, seed, n_filters_initial*(2**1), kernel_size)
    # 1 transpose convolution and concat + 2 convolutions
    pre19  = io_conv_2D_transp(conv18, n_filters_initial*(2**0), kernel_transpose_size, stride_size, seed=seed)
    up19   = io_concat_pad(pre19, conv1, 3)
    conv19 = io_conv_2D(up19, seed, n_filters_initial*(2**0), kernel_size)
    conv19 = io_conv_2D(conv19, seed, n_filters_initial*(2**0), kernel_size)
    ##################### Decoder flow ends #####################
    
     # final 1x1 convolution
    # construct model
    conv10 = io_conv_2D(conv9, seed, 1, 1)
    conv20 = io_conv_2D(conv19, seed, 3, 1)
    Output = io_concat_pad(conv10, conv20, 3)
    # construct model
    model = Model(inputs=[conv0], outputs=[Output])

    # Print info about model
    model.summary()

    # Set training parameters
    model.compile(loss=custom_loss(weight), optimizer=optimizers.Adam(lr=learning_rate, decay=0.0002))

    early = EarlyStopping(monitor  = 'val_loss',
                          mode     = 'min',
                          verbose  = 0,
                          patience = 10)
    check = ModelCheckpoint('Depth5/{}_{}_{}/best_{}_{}_{}.h5'.format(weight, n_filters_initial, seed, weight, n_filters_initial, seed),
                            monitor           = 'val_loss',
                            mode              = 'min',
                            verbose           = 0,
                            save_best_only    = True,
                            save_weights_only = False)

    # Train network
    with tf.device('/gpu:0'):
        train_model = model.fit(train_im, train_sol,
                                batch_size=batch_size, epochs=n_epochs,
                                validation_data=(valid_im, valid_sol),
                                callbacks       = [early, check])

    return(model, train_model)


### Twin_AE holds skip connecitons from shape decoder to flow decoder
def Twin_AE5(train_im,
            train_sol,
            valid_im,
            valid_sol,
            n_filters_initial,
            kernel_size,
            kernel_transpose_size,
            pool_size,
            stride_size,
            learning_rate,
            batch_size,
            n_epochs,
            height,
            width,
            n_channels,
            weight,
            seed):

    # Generate inputs
    conv0 = Input((height,width,n_channels))


    ##################### Encoder begins ##################
    # 2 convolutions + maxPool
    conv1 = io_conv_2D(conv0, seed, n_filters_initial*(2**0), kernel_size)
    conv1 = io_conv_2D(conv1, seed, n_filters_initial*(2**0), kernel_size)
    pool1 = io_maxp_2D(conv1, pool_size)
    # 2 convolutions + maxPool
    conv2 = io_conv_2D(pool1, seed, n_filters_initial*(2**1), kernel_size)
    conv2 = io_conv_2D(conv2, seed, n_filters_initial*(2**1), kernel_size)
    pool2 = io_maxp_2D(conv2, pool_size)
    # 2 convolutions + maxPool
    conv3 = io_conv_2D(pool2, seed, n_filters_initial*(2**2), kernel_size)
    conv3 = io_conv_2D(conv3, seed, n_filters_initial*(2**2), kernel_size)
    pool3 = io_maxp_2D(conv3, pool_size)
    # 2 convolutions + maxPool
    conv4 = io_conv_2D(pool3, seed, n_filters_initial*(2**3), kernel_size)
    conv4 = io_conv_2D(conv4, seed, n_filters_initial*(2**3), kernel_size)
    pool4 = io_maxp_2D(conv4, pool_size)
    # 2 convolutions
    conv5 = io_conv_2D(pool4, seed, n_filters_initial*(2**4), kernel_size)
    conv5 = io_conv_2D(conv5, seed, n_filters_initial*(2**4), kernel_size)
    ##################### Encoder ends ##################


    ##################### Decoder shape begins ##################
    # 1 transpose convolution + 2 convolutions
    pre6  = io_conv_2D_transp(conv5, n_filters_initial*(2**3), kernel_transpose_size, stride_size, seed=seed)
    up6   = io_pad(pre6, conv4)
    conv6 = io_conv_2D(up6, seed, n_filters_initial*(2**3), kernel_size)
    conv6 = io_conv_2D(conv6, seed, n_filters_initial*(2**3), kernel_size)
    # 1 transpose convolution + 2 convolutions
    pre7  = io_conv_2D_transp(conv6 , n_filters_initial*(2**2), kernel_transpose_size, stride_size, seed=seed)
    up7   = io_pad(pre7, conv3)
    conv7 = io_conv_2D(up7, seed, n_filters_initial*(2**2), kernel_size)
    conv7 = io_conv_2D(conv7, seed, n_filters_initial*(2**2), kernel_size)
    # 1 transpose convolution + 2 convolutions
    pre8  = io_conv_2D_transp(conv7, n_filters_initial*(2**1), kernel_transpose_size, stride_size, seed=seed)
    up8   = io_pad(pre8, conv2)
    conv8 = io_conv_2D(up8, seed, n_filters_initial*(2**1), kernel_size)
    conv8 = io_conv_2D(conv8, seed, n_filters_initial*(2**1), kernel_size)
    # 1 transpose convolution + 2 convolutions
    pre9  = io_conv_2D_transp(conv8, n_filters_initial*(2**0), kernel_transpose_size, stride_size, seed=seed)
    up9   = io_pad(pre9, conv1)
    conv9 = io_conv_2D(up9, seed, n_filters_initial*(2**0), kernel_size)
    conv9 = io_conv_2D(conv9, seed, n_filters_initial*(2**0), kernel_size)
    ##################### Decoder shape ends #####################


    ##################### Decoder flow begins ##################
    # 1 transpose convolution and concat + 2 convolutions
    pre16  = io_conv_2D_transp(conv5, n_filters_initial*(2**3), kernel_transpose_size, stride_size, seed=seed)
    up16   = io_concat_pad(pre16, conv6, 3)
    #up16   = io_concat_pad(pre16, up6, 3)
    conv16 = io_conv_2D(up16, seed, n_filters_initial*(2**3), kernel_size)
    conv16 = io_conv_2D(conv16, seed, n_filters_initial*(2**3), kernel_size)
    # 1 transpose convolution and concat + 2 convolutions
    pre17  = io_conv_2D_transp(conv16, n_filters_initial*(2**2), kernel_transpose_size, stride_size, seed=seed)
    up17   = io_concat_pad(pre17, conv7, 3)
    #up17   = io_concat_pad(pre17, up7, 3)
    conv17 = io_conv_2D(up17, seed, n_filters_initial*(2**2), kernel_size)
    conv17 = io_conv_2D(conv17, seed, n_filters_initial*(2**2), kernel_size)
    # 1 transpose convolution and concat + 2 convolutions
    pre18  = io_conv_2D_transp(conv17, n_filters_initial*(2**1), kernel_transpose_size, stride_size, seed=seed)
    up18   = io_concat_pad(pre18, conv8, 3)
    #up18   = io_concat_pad(pre18, up8, 3)
    conv18 = io_conv_2D(up18, seed, n_filters_initial*(2**1), kernel_size)
    conv18 = io_conv_2D(conv18, seed, n_filters_initial*(2**1), kernel_size)
    # 1 transpose convolution and concat + 2 convolutions
    pre19  = io_conv_2D_transp(conv18, n_filters_initial*(2**0), kernel_transpose_size, stride_size, seed=seed)
    up19   = io_concat_pad(pre19, conv9, 3)
    #up19   = io_concat_pad(pre19, up9, 3)
    conv19 = io_conv_2D(up19, seed, n_filters_initial*(2**0), kernel_size)
    conv19 = io_conv_2D(conv19, seed, n_filters_initial*(2**0), kernel_size)
    ##################### Decoder flow ends #####################

    # construct model
    conv10 = io_conv_2D(conv9, seed, 3, 1)
    conv20 = io_conv_2D(conv19, seed, 1, 1)
    Output = io_concat_pad(conv20, conv10, 3)
    # construct model
    model = Model(inputs=[conv0], outputs=[Output])

    # Print info about model
    model.summary()

    # Set training parameters
    model.compile(loss=custom_loss(weight), optimizer=optimizers.Adam(lr=learning_rate, decay=0.0002))

    early = EarlyStopping(monitor  = 'val_loss',
                          mode     = 'min',
                          verbose  = 0,
                          patience = 10)
    check = ModelCheckpoint('Depth5/{}_{}_{}/best_{}_{}_{}.h5'.format(weight, n_filters_initial, seed, weight, n_filters_initial, seed),
                            monitor           = 'val_loss',
                            mode              = 'min',
                            verbose           = 0,
                            save_best_only    = True,
                            save_weights_only = False)

    # Train network
    with tf.device('/gpu:0'):
        train_model = model.fit(train_im, train_sol,
                                batch_size=batch_size, epochs=n_epochs,
                                validation_data=(valid_im, valid_sol),
                                callbacks       = [early, check])

    return(model, train_model)



## Twin_AE with 6 blocks conv_conv_maxpooling blocks in the encoder
def Twin_AE6(train_im,
            train_sol,
            valid_im,
            valid_sol,
            n_filters_initial,
            kernel_size,
            kernel_transpose_size,
            pool_size,
            stride_size,
            learning_rate,
            batch_size,
            n_epochs,
            height,
            width,
            n_channels,
            weight,
            seed):

    # Generate inputs
    conv0 = Input((height,width,n_channels))
    ##################### Encoder begins ##################
    # 2 convolutions + maxPool
    conv1 = io_conv_2D(conv0, seed, n_filters_initial*(2**0), kernel_size)
    conv1 = io_conv_2D(conv1, seed, n_filters_initial*(2**0), kernel_size)
    pool1 = io_maxp_2D(conv1, pool_size)
    # 2 convolutions + maxPool
    conv2 = io_conv_2D(pool1, seed, n_filters_initial*(2**1), kernel_size)
    conv2 = io_conv_2D(conv2, seed, n_filters_initial*(2**1), kernel_size)
    pool2 = io_maxp_2D(conv2, pool_size)
    # 2 convolutions + maxPool
    conv3 = io_conv_2D(pool2, seed, n_filters_initial*(2**2), kernel_size)
    conv3 = io_conv_2D(conv3, seed, n_filters_initial*(2**2), kernel_size)
    pool3 = io_maxp_2D(conv3, pool_size)
    # 2 convolutions + maxPool
    conv4 = io_conv_2D(pool3, seed, n_filters_initial*(2**3), kernel_size)
    conv4 = io_conv_2D(conv4, seed, n_filters_initial*(2**3), kernel_size)
    pool4 = io_maxp_2D(conv4, pool_size)
    # 2 convolutions
    conv5 = io_conv_2D(pool4, seed, n_filters_initial*(2**4), kernel_size)
    conv5 = io_conv_2D(conv5, seed, n_filters_initial*(2**4), kernel_size)
    pool5 = io_maxp_2D(conv5, pool_size)
    # 2 convolutions
    conv55 = io_conv_2D(pool5, seed, n_filters_initial*(2**5), kernel_size)
    conv55 = io_conv_2D(conv55, seed, n_filters_initial*(2**5), kernel_size)
    ##################### Encoder ends ##################


    ##################### Decoder shape begins ##################
    pre66  = io_conv_2D_transp(conv55, n_filters_initial*(2**4), kernel_transpose_size, stride_size, seed=seed)
    up66   = io_pad(pre66, conv5)
    conv66 = io_conv_2D(up66, seed, n_filters_initial*(2**4), kernel_size)
    conv66 = io_conv_2D(conv66, seed, n_filters_initial*(2**4), kernel_size)

    # 1 transpose convolution + 2 convolutions
    pre6  = io_conv_2D_transp(conv66, n_filters_initial*(2**3), kernel_transpose_size, stride_size, seed=seed)
    up6   = io_pad(pre6, conv4)
    conv6 = io_conv_2D(up6, seed, n_filters_initial*(2**3), kernel_size)
    conv6 = io_conv_2D(conv6, seed, n_filters_initial*(2**3), kernel_size)
    # 1 transpose convolution + 2 convolutions
    pre7  = io_conv_2D_transp(conv6 , n_filters_initial*(2**2), kernel_transpose_size, stride_size, seed=seed)
    up7   = io_pad(pre7, conv3)
    conv7 = io_conv_2D(up7, seed, n_filters_initial*(2**2), kernel_size)
    conv7 = io_conv_2D(conv7, seed, n_filters_initial*(2**2), kernel_size)
    # 1 transpose convolution + 2 convolutions
    pre8  = io_conv_2D_transp(conv7, n_filters_initial*(2**1), kernel_transpose_size, stride_size, seed=seed)
    up8   = io_pad(pre8, conv2)
    conv8 = io_conv_2D(up8, seed, n_filters_initial*(2**1), kernel_size)
    conv8 = io_conv_2D(conv8, seed, n_filters_initial*(2**1), kernel_size)
    # 1 transpose convolution + 2 convolutions
    pre9  = io_conv_2D_transp(conv8, n_filters_initial*(2**0), kernel_transpose_size, stride_size, seed=seed)
    up9   = io_pad(pre9, conv1) 
    conv9 = io_conv_2D(up9, seed, n_filters_initial*(2**0), kernel_size)
    conv9 = io_conv_2D(conv9, seed, n_filters_initial*(2**0), kernel_size)
    ##################### Decoder shape ends #####################


    ##################### Decoder flow begins ##################
    pre166  = io_conv_2D_transp(conv55, n_filters_initial*(2**4), kernel_transpose_size, stride_size, seed=seed)
    up166   = io_pad(pre166, conv66)
    conv166 = io_conv_2D(up166, seed, n_filters_initial*(2**4), kernel_size)
    conv166 = io_conv_2D(conv166, seed, n_filters_initial*(2**4), kernel_size)

    # 1 transpose convolution and concat + 2 convolutions
    pre16  = io_conv_2D_transp(conv166, n_filters_initial*(2**3), kernel_transpose_size, stride_size, seed=seed)
    up16   = io_concat_pad(pre16, conv6, 3)
    conv16 = io_conv_2D(up16, seed, n_filters_initial*(2**3), kernel_size)
    conv16 = io_conv_2D(conv16, seed, n_filters_initial*(2**3), kernel_size)
    # 1 transpose convolution and concat + 2 convolutions
    pre17  = io_conv_2D_transp(conv16, n_filters_initial*(2**2), kernel_transpose_size, stride_size, seed=seed)
    up17   = io_concat_pad(pre17, conv7, 3)
    conv17 = io_conv_2D(up17, seed, n_filters_initial*(2**2), kernel_size)
    conv17 = io_conv_2D(conv17, seed, n_filters_initial*(2**2), kernel_size)
    # 1 transpose convolution and concat + 2 convolutions
    pre18  = io_conv_2D_transp(conv17, n_filters_initial*(2**1), kernel_transpose_size, stride_size, seed=seed)
    up18   = io_concat_pad(pre18, conv8, 3)
    conv18 = io_conv_2D(up18, seed, n_filters_initial*(2**1), kernel_size)
    conv18 = io_conv_2D(conv18, seed, n_filters_initial*(2**1), kernel_size)
    # 1 transpose convolution and concat + 2 convolutions
    pre19  = io_conv_2D_transp(conv18, n_filters_initial*(2**0), kernel_transpose_size, stride_size, seed=seed)
    up19   = io_concat_pad(pre19, conv9, 3)
    conv19 = io_conv_2D(up19, seed, n_filters_initial*(2**0), kernel_size)
    conv19 = io_conv_2D(conv19, seed, n_filters_initial*(2**0), kernel_size)
    ##################### Decoder flow ends #####################

    # final 1x1 convolution
    conv10 = io_conv_2D(conv9, seed, 1, 1)
    conv20 = io_conv_2D(conv19, seed, 3, 1)
    Output = io_concat_pad(conv10, conv20, 3)
    # construct model
    model = Model(inputs=[conv0], outputs=[Output])

    # Print info about model
    model.summary()

    # Set training parameters
    model.compile(loss=custom_loss(weight), optimizer=optimizers.Adam(lr=learning_rate, decay=0.0002))

    early = EarlyStopping(monitor  = 'val_loss',
                          mode     = 'min',
                          verbose  = 0,
                          patience = 10)
    check = ModelCheckpoint('Depth6/{}_{}_{}/best_{}_{}_{}.h5'.format(weight, n_filters_initial,  seed, weight, n_filters_initial, seed),
                            monitor           = 'val_loss',
                            mode              = 'min',
                            verbose           = 0,
                            save_best_only    = True,
                            save_weights_only = False)

    # Train network
    with tf.device('/gpu:0'):
        train_model = model.fit(train_im, train_sol,
                                batch_size=batch_size, epochs=n_epochs,
                                validation_data=(valid_im, valid_sol),
                                callbacks       = [early, check])

    return(model, train_model)

