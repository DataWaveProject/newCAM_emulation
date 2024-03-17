#### These data, mean, std values are to be passed to Model.py, where the data is first normalised and 
#### then passed to data_loader in loaddata.py and then it is passed to the model.

import numpy as np

ilev = 93
dim_NN =int(8*ilev+4)
dim_NNout =int(2*ilev)


"""Read the data and the corresponding mean and std deviation"""

"""Iterating through the data files"""
s_list = list(range(1, 6))
for iter in s_list:
    filename = "Demodata/Convection/newCAM_demo_sub_" + str(iter).zfill(1) + ".nc"  # data file
    print('working on: ', filename)
    fm = np.load('Demodata/mean_demo_sub.npz')  # mean file
    fs = np.load('Demodata/std_demo_sub.npz')   # std deviation file





def data_loader (U,V,T, DSE, NM, NETDT, Z3, RHOI, PS, lat, lon, UTGWSPEC, VTGWSPEC):
  """ Function to iterate over the data read by the above part of code"""
  Ncol = U.shape[1]
  x_train = np.vstack([
        U.reshape(-1, Ncol),
        V.reshape(-1, Ncol),
        T.reshape(-1, Ncol),
        DSE.reshape(-1, Ncol),
        NM.reshape(-1, Ncol),
        NETDT.reshape(-1, Ncol),
        Z3.reshape(-1, Ncol),
        RHOI.reshape(-1, Ncol),
        PS.reshape(1, Ncol),
        lat.reshape(1, Ncol),
        lon.reshape(1, Ncol)
    ])

  y_train = np.vstack([
        UTGWSPEC.reshape(-1, Ncol),
        VTGWSPEC.reshape(-1, Ncol)
    ])

  return x_train,y_train





