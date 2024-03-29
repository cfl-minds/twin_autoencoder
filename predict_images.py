### obtain the mean squared error or pixel-wise relative error on some shapes. 

import numpy             as np
# Custom imports
from params         import *
from datasets_utils import *


model = load_model('Depth5/best_0.1_12_2.h5', custom_objects={'tf': tf}, compile=False)
model.summary()

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

### get MSE of the test set
# e3, mse = predict_error(model, imgs, sols)
# print(np.mean(e3, axis=0))
# print(np.mean(mse, axis=0))
# np.savetxt('e3.csv', e3, delimiter=',', header='shapes,u,v,p')
# np.savetxt('mse.csv',mse, delimiter=',', header='shapes,flow')


# Make prediction and compute error on several shapes
predicted_imgs, error = predict_images(model, imgs, sols)
# Output a prediction example
for i in range(n_imgs):
     im = i
     shape = imgs[i,:,:,0]
     show_image_prediction(shape, sols[im, :, :, :], predicted_imgs[im, :, :, :],
             error[im, :, :,:], im+1, 0)
     show_image_prediction(shape, sols[im, :, :, :], predicted_imgs[im, :, :, :],
             error[im, :, :,:], im+1, 1)
     show_image_prediction(shape, sols[im, :, :, :], predicted_imgs[im, :, :, :],
             error[im, :, :,:], im+1, 2)
     show_image_prediction(shape, sols[im, :, :, :], predicted_imgs[im, :, :, :],
             error[im, :, :,:], im+1, 3)
