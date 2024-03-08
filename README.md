# newCAM-Emulation
This is a DNN written with PyTorch to Emulate the gravity wave drag (GWD, both zonal and meridional ) in the WACCM Simulation.


# DemoData
Sample output data from CAM.
It is 3D global output from the mid-top CAM model, on the original model grid.

However, the demo data here is one very small part of the CAM output due to storage limit of Github. NN trained on this Demodata will not work.

# Installing

Clone this repo and enter it.\
Then run:
```
pip install .
```
to install the neccessary dependencies.\
It is recommended this is done from inside a virtual environment.

# data loader
load 3D CAM data and reshaping them to the NN input.

# Using a FNN to train and predict the GWD
train.py train the files and generate the weights for NN.

NN-pred.py load the weights and do prediction.

# Coupling ? future work
replace original GWD scheme in WACCM with this emulator.

a. the emulator can be trained offline

b. training the emulator online


