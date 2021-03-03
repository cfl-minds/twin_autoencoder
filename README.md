# twin_decoder
This is the repository for the article "A twin-decoder structure for incompressible laminar flow reconstruction with uncertainty estimation around 2D obstacles".![architecture](./images/architecture.png)

The entire project are validated in **Ubuntu 20.04**.
## Package requisitions
- **Python** = 3.6.9
- **Tensorflow** = 2.0

## Structure of the repository
- **dataset_utils** : general functions, including data pre-processing, results plotting etc..
- **network_utils** : layers and models used in the article;
- **params** : directions, network hyper-parameters etc..
- **train_predict** : load data set and train a model, save the learning history and the mean squared error over the whole data set;
- **mistake_minimization** : an independent script find the optimal threshold of the reconstruction error for the qualitative method;
- **nll_minimization** : an independent script solving problem (10) for the quantitative trust level method;
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
To get the reconstruction threshold, run
```
python3 mistake_minimization.py
```
![qualitative](./images/qualitative.png)


To get the regression line, run
```
python3 nll_minimization.py
```
![quantitative](./images/quantitative.png)
