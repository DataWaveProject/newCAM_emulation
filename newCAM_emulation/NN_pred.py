"""Prediction module for the neural network."""

import matplotlib.pyplot as plt
import Model
import netCDF4 as nc
import numpy as np
import torch
import torch.nn.functional as nnF
import torchvision
from loaddata import data_loader, newnorm
from savedata import save_netcdf_file
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

"""
Determine if any GPUs are available
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


"""
Initialize Hyperparameters
"""
ilev = 93
ilev_94 = 94

dim_NN = 8 * ilev + 4
dim_NNout = 2 * ilev

batch_size = 8
learning_rate = 1e-4
num_epochs = 1


## load mean and std for normalization
fm = np.load("Demodata/mean_demo_sub.npz")
fs = np.load("Demodata/std_demo_sub.npz")

Um = fm["U"]
Vm = fm["V"]
Tm = fm["T"]
DSEm = fm["DSE"]
NMm = fm["NM"]
NETDTm = fm["NETDT"]
Z3m = fm["Z3"]
RHOIm = fm["RHOI"]
PSm = fm["PS"]
latm = fm["lat"]
lonm = fm["lon"]
UTGWSPECm = fm["UTGWSPEC"]
VTGWSPECm = fm["VTGWSPEC"]

Us = fs["U"]
Vs = fs["V"]
Ts = fs["T"]
DSEs = fs["DSE"]
NMs = fs["NM"]
NETDTs = fs["NETDT"]
Z3s = fs["Z3"]
RHOIs = fs["RHOI"]
PSs = fs["PS"]
lats = fs["lat"]
lons = fs["lon"]
UTGWSPECs = fs["UTGWSPEC"]
VTGWSPECs = fs["VTGWSPEC"]


"""
Initialize the network and the Adam optimizer
"""
GWnet = Model.FullyConnected()

optimizer = torch.optim.Adam(GWnet.parameters(), lr=learning_rate)


s_list = list(range(6, 7))

data_vars = []

for iter in s_list:
    if iter > 0:
        GWnet.load_state_dict(torch.load("./conv_torch.pth"))
        GWnet.eval()
    print("data loader iteration", iter)
    filename = "Demodata/newCAM_demo_sub_" + str(iter).zfill(1) + ".nc"

    F = nc.Dataset(filename)
    PS = np.asarray(F["PS"][0, :])
    PS = newnorm(PS, PSm, PSs)

    Z3 = np.asarray(F["Z3"][0, :, :])
    Z3 = newnorm(Z3, Z3m, Z3s)

    U = np.asarray(F["U"][0, :, :])
    U = newnorm(U, Um, Us)

    V = np.asarray(F["V"][0, :, :])
    V = newnorm(V, Vm, Vs)

    T = np.asarray(F["T"][0, :, :])
    T = newnorm(T, Tm, Ts)

    lat = F["lat"]
    lat = newnorm(lat, np.mean(lat), np.std(lat))

    lon = F["lon"]
    lon = newnorm(lon, np.mean(lon), np.std(lon))

    DSE = np.asarray(F["DSE"][0, :, :])
    DSE = newnorm(DSE, DSEm, DSEs)

    RHOI = np.asarray(F["RHOI"][0, :, :])
    RHOI = newnorm(RHOI, RHOIm, RHOIs)

    NETDT = np.asarray(F["NETDT"][0, :, :])
    NETDT = newnorm(NETDT, NETDTm, NETDTs)

    NM = np.asarray(F["NMBV"][0, :, :])
    NM = newnorm(NM, NMm, NMs)

    UTGWSPEC = np.asarray(F["UTGWSPEC"][0, :, :])
    UTGWSPEC = newnorm(UTGWSPEC, UTGWSPECm, UTGWSPECs)

    VTGWSPEC = np.asarray(F["VTGWSPEC"][0, :, :])
    VTGWSPEC = newnorm(VTGWSPEC, VTGWSPECm, VTGWSPECs)

    print("shape of PS", np.shape(PS))
    print("shape of Z3", np.shape(Z3))
    print("shape of U", np.shape(U))
    print("shape of V", np.shape(V))
    print("shape of T", np.shape(T))
    print("shape of DSE", np.shape(DSE))
    print("shape of RHOI", np.shape(RHOI))
    print("shape of NETDT", np.shape(NETDT))
    print("shape of NM", np.shape(NM))
    print("shape of UTGWSPEC", np.shape(UTGWSPEC))
    print("shape of VTGWSPEC", np.shape(VTGWSPEC))

    output_filename = f"Demodata/normalized_data_{iter}.nc"
    save_netcdf_file(
        output_filename,
        lat,
        lon,
        ilev,
        ilev_94,
        PS,
        Z3,
        U,
        V,
        T,
        DSE,
        RHOI,
        NETDT,
        NM,
        UTGWSPEC,
        VTGWSPEC,
    )

    x_test, y_test = data_loader(
        U, V, T, DSE, NM, NETDT, Z3, RHOI, PS, lat, lon, UTGWSPEC, VTGWSPEC
    )

    print("shape of x_test", np.shape(x_test))
    print("shape of y_test", np.shape(y_test))
    data = Model.myDataset(X=x_test, Y=y_test)
    test_loader = DataLoader(data, batch_size=len(data), shuffle=False)
    print(test_loader)

    for batch, (X, Y) in enumerate(test_loader):
        print(np.shape(Y))
        pred = GWnet(X)
        truth = Y.cpu().detach().numpy()
        predict = pred.cpu().detach().numpy()

    print(np.corrcoef(truth.flatten(), predict.flatten())[0, 1])
    print("shape of truth ", np.shape(truth))
    print("shape of prediction", np.shape(predict))

    # np.save("./pred_data_" + str(iter) + ".npy", predict)
    output_filename = f"Demodata/predicted_data_{iter}.nc"
    save_netcdf_file(
        output_filename,
        lat,
        lon,
        ilev,
        ilev_94,
        PS,
        Z3,
        U,
        V,
        T,
        DSE,
        RHOI,
        NETDT,
        NM,
        UTGWSPEC,
        VTGWSPEC,
    )
