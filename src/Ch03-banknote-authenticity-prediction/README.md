## Project Summary

This repository implements a 4-Layer neural network from scratch to classify whether a banknote is authentic or not. The data used in this project was initially taken from [here](http://archive.ics.uci.edu/dataset/267/banknote+authentication).

## Setup the Environment

* Create a virtualenv with Python and activate it.

```bash
# In MAC
python3 -m venv .banknote
source .banknote/bin/activate

# In Windows
python3 -m venv .banknote
source .banknote/Scripts/activate
```
* Run `pip install -r requirements.txt` to install the necessary dependencies

## Download the Dataset
```bash
# using curl
curl http://archive.ics.uci.edu/static/public/267/banknote+authentication.zip -o dataset.zip

# using wget
wget -O dataset.zip http://archive.ics.uci.edu/static/public/267/banknote+authentication.zip
```

Make sure to unzip the file in /raw-dataset folder
### Dataset Description
The dataset used in this project consists of csv file containing a set of banknotes. Each banknote is represented with four features that were extracted from the original image of the banknote using wavelet transform.


## Folders and Files Description

1. clean-dataset: Contains the dataset after being preprocessed
2. data\_exploration.ipynb: normalizes the dataset and plot its features.
3. utils.py: Contains the implementation of the necessary classes used in building the neural network.
4. neural\_net.ipynb: trains and evaluates the neural network
6. requirements.txt: Contains the set of dependencies that are necessary for the app