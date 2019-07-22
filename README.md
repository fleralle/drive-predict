# Drive & Predict

[Photo by Pang Yuhao on Unsplash](images/pang-yuhao-X00ZafKUdBo-unsplash.jpg)Photo by Pang Yuhao on Unsplash

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
lib under `models/data.py` containing event (such as brake, start, turn) extraction
functions and also functions that calculate metrics around those events.

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
