# Imports
import os
import logging
import tensorflow as tf
import numpy      as np
import random     as rn
import keras      as k


# Data directories
dataset_dir           = 'Data'
shape_dir             = dataset_dir+'/shapes'
sdf_dir             = dataset_dir+'/sdf_nb'
u_dir                 = dataset_dir+'/unb'
v_dir                 = dataset_dir+'/vnb'
p_dir                 = dataset_dir+'/pnb'
shape_dir_test        = dataset_dir+'/shapes_test'
u_dir_test            = dataset_dir+'/u_test'
v_dir_test            = dataset_dir+'/v_test'
p_dir_test            = dataset_dir+'/p_test'

#input_dir             = shape_dir
#sol_dir               = u_dir
#input_dir_test        = shape_dir_test
#sol_dir_test          = velocity_dir_test
# Model directories
model_dir             = './'
model_h5              = model_dir+'best_0.1.h5'
model_json            = model_dir+'model.json'

# Image data
img_width             = 1500
img_height            = 1000
downscaling           = 10
color                 = 'bw'

# Dataset data
train_size            = 0.8
valid_size            = 0.1
tests_size            = 0.1

# Network data
n_filters_initial     = 16
kernel_size           = 3
kernel_transpose_size = 2
stride_size           = 2
pool_size             = 2

# Learning data
learning_rate         = 1.0e-3
batch_size            = 128
n_epochs              = 1000
network               = 'U_net'

# Hardware
train_with_gpu    = True

# Set tf verbosity
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
logger                             = tf.get_logger()
logger.setLevel(logging.ERROR)

# Set random seeds
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '1'
tf.random.set_seed(1)
np.random.seed(1)
rn.seed(1)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True


tf.compat.v1.set_random_seed(1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),
                            config=session_conf)


tf.compat.v1.keras.backend.set_session(sess)

