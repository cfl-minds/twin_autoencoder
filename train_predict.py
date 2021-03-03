# Generic imports
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import keras.backend     as K
from   os                import mkdir

# Custom imports
from params         import *
from datasets_utils import *
from networks_utils import *

# Handle GPUs
if (train_with_gpu):
    # os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Show the devices in use
print('########### Training flow prediction network ###########')
print('')
print("Devices in use:")
cpus = tf.config.experimental.list_physical_devices('CPU')
for cpu in cpus:
    print("Name:", cpu.name, "  Type:", cpu.device_type)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)
print('')


## Set which GPU to use out of two V100 Teslas
GPU_to_use = 0 #1

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first or second GPU
    useGPU = GPU_to_use
    try:
        tf.config.experimental.set_visible_devices(gpus[useGPU], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        tf.config.experimental.set_memory_growth(gpus[useGPU], True)
    except RuntimeError as e:
        print(" Visible devices must be set before GPUs have been initialized")
        print(e)



# Load image dataset
imgs, n_imgs, height, width, n_channels  = load_img_dataset(shape_dir,
                                                           downscaling,
                                                           color)


# Load solutions dataset
u_sols, n_sols, height, width, n_channels = load_img_dataset(u_dir,
                                                             downscaling,
                                                             color)

v_sols, n_sols, height, width, n_channels = load_img_dataset(v_dir,  
                                                             downscaling,
                                                             color)

p_sols, n_sols, height, width, n_channels = load_img_dataset(p_dir,
                                                             downscaling,
                                                             color)



sols = np.concatenate((imgs, u_sols, v_sols, p_sols), axis=-1)
del u_sols, v_sols, p_sols

# Split data into training, validation and testing sets
(imgs_train,
imgs_valid,
imgs_tests) = split_dataset(imgs, train_size, valid_size, tests_size)
del imgs

(sols_train,
sols_valid,
sols_tests) = split_dataset(sols, train_size, valid_size, tests_size)
del sols

    # Print informations
print('Training   set size is', imgs_train.shape[0])
print('Validation set size is', imgs_valid.shape[0])
print('Test       set size is', imgs_tests.shape[0]) 
print('Input images downscaled to',str(width)+'x'+str(height))

for n_filters_initial in [12]:
	for weight in [0.1]
		for seed in [10]:
			results_dir = 'Depth5/{}_{}_{}'.format(weight, n_filters_initial, seed)
			try:
				mkdir(results_dir)
			except:
				pass
			start              = time.time()
			model, train_model = Twin_AE5(imgs_train,
						       sols_train,
						       imgs_valid,
						       sols_valid,
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
						       seed)
			end                = time.time()
			print('Training model {}_{}_{} time was '.format(weight, n_filters_initial, seed),end-start,' seconds')
			min_valid = plot_accuracy_and_loss(train_model, results_dir)
			print('The minimum validation error is ', min_valid)

			model     = load_model(results_dir + '/best_' + '{}_{}_{}'.format(weight, n_filters_initial, seed) + '.h5', compile=False)
			# Make prediction and compute error
			
			start     = time.time()
			mse_train = predict_images(model, imgs_train, sols_train)
			pd.DataFrame(mse_train).to_csv(results_dir + '/mse_train.csv')
			mse_valid = predict_images(model, imgs_valid, sols_valid)
			pd.DataFrame(mse_valid).to_csv(results_dir + '/mse_valid.csv')
			mse_test  = predict_images(model, imgs_tests, sols_tests)
			pd.DataFrame(mse_test).to_csv(results_dir + '/mse_test.csv')

			end       = time.time()

			print('Model', results_dir, ' all results saved in ', end-start, ' seconds')

