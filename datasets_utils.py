# Generic imports
import os
import re
import sys
import time
import math
import numpy as np
import matplotlib
if (sys.platform == 'darwin'):
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Imports with probable installation required
try:
    import skimage
except ImportError:
    print('*** Missing required packages, I will install them for you ***')
    os.system('pip3 install scikit-image')
    import skimage

try:
    import keras
except ImportError:
    print('*** Missing required packages, I will install them for you ***')
    os.system('pip3 install keras')
    import keras

try:
    import progress.bar
except ImportError:
    print('*** Missing required packages, I will install them for you ***')
    os.system('pip3 install progress')
    import progress.bar

from keras.utils               import plot_model
from keras.models              import load_model
from keras.preprocessing.image import img_to_array, load_img

### ************************************************
### Split dataset in training, validation and tests
def split_dataset(dataset, train_size, valid_size, tests_size):
    # Check sizes
    if ((train_size + valid_size + tests_size) != 1.0):
        print('Error in split_dataset')
        print('The sum of the three provided sizes must be 1.0')
        exit()

    # Compute sizes
    n_data     = dataset.shape[0]
    train_size = math.floor(n_data*train_size)
    valid_size = math.floor(n_data*valid_size) + train_size
    tests_size = math.floor(n_data*tests_size) + valid_size

    # Split
    if (dataset.ndim == 1):
        (dataset_train,
         dataset_valid,
         dataset_tests) = (dataset[0:train_size],
                           dataset[train_size:valid_size],
                           dataset[valid_size:])

    if (dataset.ndim == 2):
        (dataset_train,
         dataset_valid,
         dataset_tests) = (dataset[0:train_size,         :],
                           dataset[train_size:valid_size,:],
                           dataset[valid_size:,          :])

    if (dataset.ndim == 3):
        (dataset_train,
         dataset_valid,
         dataset_tests) = (dataset[0:train_size,         :,:],
                           dataset[train_size:valid_size,:,:],
                           dataset[valid_size:,          :,:])

    if (dataset.ndim == 4):
        (dataset_train,
         dataset_valid,
         dataset_tests) = (dataset[0:train_size,         :,:,:],
                           dataset[train_size:valid_size,:,:,:],
                           dataset[valid_size:,          :,:,:])

    return dataset_train, dataset_valid, dataset_tests

### ************************************************
### Load image
def get_img(img_name):
    x = img_to_array(load_img(img_name))

    return x

### ************************************************
### Load and reshape image
def load_and_reshape_img(img_name, height, width, color):
    # Load and reshape
    x = img_to_array(load_img(img_name))
    x = skimage.transform.resize(x,(height,width),
                                 anti_aliasing=True,
                                 #preserve_range=True,
                                 #order=0,
                                 mode='constant')

    # Handle color
    if (color == 'bw'):
        x = (x[:,:,0] + x[:,:,1] + x[:,:,2])/3.0
        x = x[:,:,np.newaxis]

    # Rescale
    x = x.astype('float16')/255

    return x

### ************************************************
### Load full image dataset
def load_img_dataset(my_dir, downscaling, color):
    # Start counting time
    start = time.time()
    
    # Count files in directory
    data_files = [f for f in os.listdir(my_dir) if (f[0:5] == 'shape')]
    data_files = sorted(data_files)
    n_imgs     = math.floor(len(data_files))
    print('I found {} images'.format(n_imgs))

    # Check size of first image
    img    = get_img(my_dir+'/'+data_files[0])
    height = img.shape[0]
    width  = img.shape[1]

    # Declare n_channels
    if (color == 'bw'):  n_channels = 1
    if (color == 'rgb'): n_channels = 3

    # Compute downscaling and allocate array
    height = math.floor(height/downscaling)
    width  = math.floor(width /downscaling)
    imgs   = np.zeros([n_imgs,height,width,n_channels])

    # Load all images
    bar = progress.bar.Bar('Loading images from '+my_dir, max=n_imgs)
    for i in range(0, n_imgs):
        imgs[i,:,:,:] = load_and_reshape_img(my_dir+'/'+data_files[i],
                                             height, width, color)
        bar.next()
    bar.finish()

    # Stop counting time
    end = time.time()
    print('Loaded ',n_imgs,' imgs in ',end-start,' seconds')

    return imgs, n_imgs, height, width, n_channels

### ************************************************
### Plot relative errors
def plot_relative_errors(predict, error, filename):
    save = np.transpose(predict[:,0])
    if (predict.shape[1] == 2):
        save = np.column_stack((save, np.transpose(predict[:,1])))
    save = np.column_stack((save, np.transpose(error[:,0])))
    if (error.shape[1] == 2):
        save = np.column_stack((save, np.transpose(error[:,1])))

    np.savetxt(filename, save)

    plt.scatter(predict[:,0],error[:,0],c=error[:,0],s=50,cmap='viridis')
    plt.colorbar()
    plt.savefig(filename+'.png')
    plt.show()

### ************************************************
### Plot accuracy and loss as a function of epochs
def plot_accuracy_and_loss(train_model, direction):
    hist       = train_model.history
    train_loss = hist['loss']
    valid_loss = hist['val_loss']
    epochs     = range(len(train_loss))
    np.savetxt(direction + '/loss',np.transpose([epochs,train_loss,valid_loss]))

    plt.semilogy(epochs, train_loss, 'g', label='Training loss')
    plt.semilogy(epochs, valid_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(direction + '/loss.png')
    plt.show()
    plt.close()

    return np.min(valid_loss)

### ************************************************
### Predict images from model and compute errors

def predict_error(model, imgs, sols):
    # Get img shape
    h = sols.shape[1]
    w = sols.shape[2]
    c = sols.shape[3]

    # Various stuff
    n_imgs    = len(imgs)
    predict   = np.zeros([n_imgs,h,w,c],dtype=np.float16)
    rel_err   = np.zeros((n_imgs,4),dtype=np.float16)
    mse       = np.zeros((n_imgs,2),dtype=np.float16)

    # Predict
    for i in range(0, n_imgs):
        img                = imgs[i,:,:,:]
        img                = img.reshape(1,h,w,1)
        predict[i,:,:,:]   = model.predict(img)
        error              = np.abs(predict[i,:,:,:]-sols[i,:,:,:])
        
        rel_err[i, :]      = np.mean(error[35:65,35:80,:] / (sols[i,35:65,35:80,:] + 1e-3), axis=(0,1))
        mse[i, :]          = [np.mean(np.square(error[i,:,:,0])), np.mean(np.square(error[i,:,:,1:]))]
    return rel_err , mse

def predict_images(model, imgs, sols):
    # Get img shape
    h = sols.shape[1]
    w = sols.shape[2]
    c = sols.shape[3]

    # Various stuff
    n_imgs    = len(imgs)
    predict   = np.zeros([n_imgs,h,w,c],dtype=np.float32)
    error     = np.zeros([n_imgs,h,w,c],dtype=np.float32)

    # Predict
    for i in range(0, n_imgs):
        img                = imgs[i,:,:,:]
        img                = img.reshape(1,h,w,1)
        predict[i,:,:,:]   = model.predict(img)
        error[i,:,:,:]      = np.abs(predict[i,:,:,:]-sols[i,:,:,:])

    return predict, error

### ************************************************
### Show an image prediction along with exact image and error
def show_image_prediction(shape, ref_img, predicted_img, error_img, i, channel):
    channels = ['shape', 'u', 'v', 'p']

    if channel == 0:
        filename = 'predictions/predicted_shape_{}'.format(i)

        fig, ax = plt.subplots()
        ax = plt.axes([0,0,1,1])
        plt.imshow(shape[:,:], interpolation='spline16', cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(filename+'_ref.png', dpi=300, bbox_inches='tight',pad_inches = 0)
        plt.close()

        fig, ax = plt.subplots()
        ax = plt.axes([0,0,1,1])
        plt.imshow(predicted_img[:,:,channel], interpolation='spline16', cmap='gray',vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(filename+'_pred.png', dpi=300, bbox_inches='tight',pad_inches = 0)
        plt.close()
    
        fig, ax = plt.subplots()
        ax = plt.axes([0,0,1,1])
        #plt.imshow(error_img[:,:,channel]/np.amax(error_img[:,:,channel]), interpolation='spline16', cmap='gray')
        plt.imshow(error_img[:,:,channel], interpolation='spline16', cmap='gray')
        print(np.max(error_img[:,:,channel]))
        plt.axis('off')
        plt.savefig(filename+'_error.png', dpi=300, bbox_inches='tight',pad_inches = 0)
        plt.close()

    else:
        filename = 'predictions/predicted_'+channels[channel]+'{}'.format(i)
        #filename = 'predictions/' +sorted(os.listdir('outliers/Data/shapes'))[i][:-4]+'_'+channels[channel] + '_'

        cmap = plt.cm.RdBu_r
        
        fig, ax = plt.subplots()
        ax = plt.axes([0,0,1,1])
        plt.imshow(ref_img[:,:,channel], interpolation='spline16', cmap=cmap, vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(filename+'_ref.png', dpi=300, bbox_inches='tight',pad_inches = 0)
        plt.close()

        fig, ax = plt.subplots()
        ax = plt.axes([0,0,1,1])
        plt.imshow(predicted_img[:,:,channel], interpolation='spline16', cmap=cmap,vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(filename+'_pred.png', dpi=300, bbox_inches='tight',pad_inches = 0)
        plt.close()
        
        fig, ax = plt.subplots()
        ax = plt.axes([0,0,1,1])
        #plt.imshow(error_img[:,:,channel]/np.amax(error_img[:,:,channel]), interpolation='spline16', cmap='gray')
        plt.imshow(error_img[:,:,channel], interpolation='spline16', cmap='RdBu_r')
        print(np.max(error_img[:,:,channel]))
        plt.axis('off')
        plt.savefig(filename+'_error.png', dpi=300, bbox_inches='tight',pad_inches = 0)
        plt.close()


