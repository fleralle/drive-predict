# Drive & Predict

![Photo by Pang Yuhao on Unsplash](images/pang-yuhao-X00ZafKUdBo-unsplash.jpg)Photo by Pang Yuhao on Unsplash

## Description
The study aims to predict distinct metrics and categories out of driving data.
How good will it be to tell you if you are a good or bad driver based on your
driving data ? Or may be, to tell the drivers to slow down during turns or increase
their braking distance.

Such model might have multiple applications like insurance rating/pricing,
driver rating system, on board warning system. Some companies (Uber, Lyft,
Progressive) are already using their own private data to predict/build their
models.

As the study involves to get lots of data around driving car, it has been
difficult to find out available datasets. The first goal of this project was to
build the app that allows data collection in order to build the model. As I ran
out of time I haven't been able to build it unfortunately. I hope I will do it
in the near future.

During this study, we will split our work into small feasible tasks. We will look
at behaviours like braking, starts, turning. To do so, I have built a data utils
lib under `models/data.py` containing event (such as braking, acceleration, turning)
extraction functions and also functions that calculate metrics around those events. The rest of the code can be found under the `models` module where there is more functionalities.

The main Jupyter notebook is available at `notebook.ipynb` [link](https://github.com/fleralle/drive-predict/blob/master/notebook.ipynb). It details every single steps taken during the project.

## Installation
In order to run the Jupyter Notebook, you will need first to download the
following dataset https://www.kaggle.com/vitorrf/cartripsdatamining/downloads/cartripsdatamining.zip/1 under the data project folder (dataset not included in
the repo as it is too big).

## Dataset Description
The original dataset is a zip file containing 38 CSV files corresponding to 38
car trips (~30 mins each)

Contents of CSV files:
* Column 1: Time (in seconds)
* Column 2: Vehicle’s speed (in m/s)
* Column 3: Shift number (0 = intermediate position)
* Column 4: Engine Load (% of max power)
* Column 5: Total Acceleration (m/s^2)
* Column 6: Engine RPM
* Column 7: Pitch
* Column 8: Lateral Acceleration (m/s^2)
* Column 9: Passenger count (0 - 5)
* Column 10: Car’s load (0 - 10)
* Column 11: Air conditioning status (0 - 4)
* Column 12: Window opening (0 - 10)
* Column 13: Radio volume (0 - 10)
* Column 14: Rain intensity (0 - 10)
* Column 15: Visibility (0 - 10)
* Column 16: Driver’s wellbeing (0 - 10)
* Column 17: Driver’s rush (0 - 10)

## Exploratory Data Analysis

The EDA has been conducted in 3 distinct steps :
* Loading the datasets and checking the structure, feature types and null values.
* Looking at feature distribution (looking for outliers) and any feature correlations.
* Extracting the driving events (braking, acceleration, turning), calculating the metrics around those events and then plotting the metrics against with our target value `driver_rush`

From the 3 events types the most notable one is the braking. Even if the numbers are not huge, the ratio of harsh braking over total braking events is five times higher when the driver is in a rush compared to not in a rush. The others events (accelerations and turnings) didn't really show a difference between `rush` vs `not rush` when looking at harsh accelerations ratio and harsh turning ratio.

The EDA highlights also the fact that the observations are not conclusives. It would have been helpful to get more data and especially more data coming from distinct drivers. Here we only have the driving measurements for one driver-one car which obviously is not ideal.

## Modelling

This step consists in trying to find the best model which can predict our target value `driver_rush`. 
We took a pragmatic approach to the modelling phase and followed the steps bellow :

* Select list of known classification models
* Run a baseline model for each of our pre-selected models
* Pick the top models base on **precision**, **recall**, **f1-score** scores
* Tune the hyper-parameters of the top model and watch score increase

The list of pre-selected classification models is :

* Logistic Regression
* Decision Tree
* Random Forest
* AdaBoost
* K Nearest Neighbours
* XGBoost
* SVC (Support Vector Classification)

Out of our baseline modelling simulations the best-performing models are XGBoost Classifier, Decision Tree Classifier, Logistic Regression and random Forest. After dealing with our imbalance targets dataset and reducing the classification task to a binary classification, it comes up that our top 2 performers were Random Forest and XGBoost Classifier.

For each top performer classifiers, a grid-search along with cross validation was run to determine the best hyper-parameters for the given classifier. 

At the end, the best performing classifier was the Random Forest with overall mean validation score of 0.707

## Conclusion

* Random Forest was the best-performing classifier to predict the driver rush indicator.
* Surprisingly accelerations and turning events didn't really indicate that it was possible to predict our target values from. This observation definitively needs more studies as we have a lack of data for now.
* Next steps include gathering more data from different diver and a way to label those data. I would look at building an app that does that and ask friends for their participation.
* Also it would be interesting to train a Neural Network and see if it can perform better than Random Forest.