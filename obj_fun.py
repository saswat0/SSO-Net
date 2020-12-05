import numpy as np
from net.CNN_model import CNN_model


def obj_fun(soln,Input, Tr, Target):
    if soln.ndim == 2:
        dim = soln.shape[1]
        v = soln.shape[0]
        fitn = np.zeros((soln.shape[0], 1))
    else:
        dim = soln.shape[0]; v = 1
        fitn = np.zeros((1, 1))

    for i in range(v):
        soln = np.array(soln)

        if soln.ndim == 2:
            sol = np.round(soln[i,:])
        else:
            sol = np.round(soln)
        sol = sol.astype(int)
        new_feat = Input[:,sol[0:dim-2]]

        Eval = CNN_model(new_feat, Target, Tr, 30, sol[dim-1])
        fitn[i] = 1 / Eval[5]
    return fitn
