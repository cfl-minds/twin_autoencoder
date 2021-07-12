# twin_decoder
This is the repository for the article "[A twin-decoder structure for incompressible laminar flow reconstruction with uncertainty estimation around 2D obstacles](https://arxiv.org/abs/2104.03619/)"
![architecture](./images/architecture.png)


The proposed CNN architecture has two decoder branches. It reconstructs the input geometry and do laminar flow prediction at the same time. The shared latent features and skip connections between the two branches make their outputs highly correlated. One can thus estimate the uncertainty of flow prediction based on the geometry reconstruction.

The data set contains 12000 random 2D obstacles, together with the laminar velocity and pressure field. It was first used in the article by J. Viquerat  and E. Hachem "[A supervised neural network for drag prediction of arbitrary 2D shapes in laminar flows at low Reynolds number](https://github.com/jviquerat/cnn_drag_prediction)"

The entire project are has been validated in **Ubuntu 20.04**. To reproduce the results, it is preferred to creat a virtual environment with **python==3.6.9**, and install the packages listed in **requirements.txt**.

## Structure of the repository
- **dataset_utils** : general functions, including data pre-processing, results plotting etc..
- **network_utils** : layers and models used in the article;
- **params** : directions, network hyper-parameters etc..
- **train_predict** : load data set and train a model, save the learning history and the mean squared error over the whole data set;
- **predict_images**: visualization of the reconstructed input and the flow prediction;
- **mistake_minimization** : an independent script searching the optimal threshold of the reconstruction error for the qualitative method (section 3.3.1);
- **nll_minimization** : an independent script solving the linear regression problem for the quantitative trust level method (section 3.3.2);
- **Depth5** : results direction saving the model, and the mean squared errors in csv format.
## Model training
Specify model's name in ```train_predict.py``` by choosing one of the four models:
```
Twin_AE5, Twin_AE6, Dual_AE, U_Dual_AE
```
To train a model, run
```
python3 train_predict.py
```
To get the threshold in the qualitative method, run
```
python3 mistake_minimization.py
```
![qualitative](./images/qualitative.png)


To get the regression line in the quantitative method, run
```
python3 nll_minimization.py
```
![quantitative](./images/quantitative.png)
