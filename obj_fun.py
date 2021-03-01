import logging
from tqdm import tqdm
import numpy as np
from net.CNN_model import CNN_model

# soln: init_soln
# Input: Feat
# Tr: Tr
# Target: Tar
logging.basicConfig(filename='SSO.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Objective Function Invoked')

def obj_fun(soln, Input, Tr, Target):
    logging.info('Soln: {0}\nInput/Feat: {1}\nTr: {2}\nTarget: {3}'.format(soln, Input, Tr, Target))
    if soln.ndim == 2:
        dim = soln.shape[1]
        v = soln.shape[0]
        fitn = np.zeros((soln.shape[0], 1))
    else:
        dim = soln.shape[0]; v = 1
        fitn = np.zeros((1, 1))

    logging.info('fitn: {0}\nv: {1}\ndim: {2}'.format(fitn, v, dim))

    for i in tqdm(range(v)):
        logging.info('Iteration {0} inside obj_fun'.format(i))
        soln = np.array(soln)

        if soln.ndim == 2:
            sol = np.round(soln[i, : ])
        else:
            sol = np.round(soln)
        sol = sol.astype(int)
        new_feat = Input[:,sol[0 : dim-2]]
        logging.info('New Feat shape: {0}\nNew Feat: {1}'.format(new_feat.shape, new_feat))
        logging.info('CNN Model Invoked')

        Eval = CNN_model(new_feat, Target, Tr, 70, sol[dim-1], indd=2)
        # Eval = CNN_model(new_feat, Tr, Target, 30, sol[dim-1], indd=2)
        fitn[i] = 1 / Eval[5]
        logging.info('Fitn value in {0} iteration: {1}'.format(i, fitn[i]))
    logging.info('Fitn computed by obj_fun: {0}'.format(fitn))
    logging.info('Exiting Objective Function')
    return fitn
