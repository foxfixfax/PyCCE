import pandas as pd
def tocsv(timespace, data, r_bath, r_dipole, order, pdkwargs={}, **kwargs):
    dshape = data.shape
    if len(dshape) == 1:
        s = pd.Series(data, index=timespace)
    elif len(dshape) == 2:
        pass
    elif len(dshape) == 3:
        data = data.reshape(data.shape[0], -1)
        projections = 1
    return 0