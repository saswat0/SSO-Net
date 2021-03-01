import logging
import numpy as np
import time
from obj_fun import obj_fun

logging.basicConfig(filename='SSO.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Inside SSO Function')

def SSO(val, objfun, x_min, x_max, kmax, Feat, Tar, Tr):
    N, D = val.shape # N speakers D features
    logging.info('Val Shape: {0}\nNumber of speakers: {1}\nNumber of Features: {2}'.format(val.shape, N, D))
    vel = np.random.rand(N, D)
    logging.info('Vel shape: {0}\nVel array (initialised randomly): {1}'.format(vel.shape, vel))
    
    alphak = np.random.rand()
    betak = np.random.rand()
    logging.info('Alphak: {0}\tBetak: {1}\t (Random Initialisation)')

    fbst = np.random.rand(kmax)
    best_sub = np.random.rand(kmax, N)
    logging.info('fbst shape: {0}\nbest_sub.shape: {1}\nInitialised Randomly'.format(fbst.shape, best_sub.shape))

    t = 1       # time interval stage of k
    gk = np.random.rand()
    x = val.copy()
    logging.info('gk value: {0}'.format(gk))

    # f = np.apply_along_axis(objfun, 0, x)
    # f = np.amax(x, 0)
    logging.info('Calling obj_fun')
    f = obj_fun(x, Feat, Tr, Tar)
    logging.info('F value returned by obj_fun'.format(f))

    # find global best and particle best
    fgbest = np.amin(f)
    igbest = np.where(f == fgbest)[0][0].item()
    logging.info('Fgbest and Igbest are derived from f returned by obj_fun')
    logging.info('Fgbest: {0}\nIgbest: {1}'.format(fgbest, igbest))
    
    gbest = x[igbest,:]
    pbest = x
    fpbest = f
    tic = time.time()
    logging.info('gbest: {0}\npbest: {1}\nfpbest: {2}'.format(gbest, pbest, fpbest))

    for k in range(kmax):
        logging.info('Iteration: {0}'.format(k))
        for i in range(N):
            for j in range(D):
                logging.info('K = {0}\tI = {1}\tJ = {2}'.format(k, i, j))
                r1 = np.random.rand()
                logging.info('r1: {0}'.format(r1))
                delf = abs(fgbest - f[i]) / fgbest
                term1 = gk * r1 * delf
                logging.info('delf: {0}\nterm1: {1}'.format(delf, term1))

                if k == 1:
                    vel[i, j] = term1
                else:
                    term2 = min(abs(term1)+alphak*r1*vel[i, j], abs(betak*vel[i, j]))
                    vel[i, j] = term2
            logging.info('Vel: {0}'.format(vel))
            rand = np.random.rand()
            if rand < 0.5:
                x[i,:] = x[i,:] + vel[i,:] * t          # forward movement update process by Eq. (10.9)
            else:
                r3 = 2 * rand - 1                       # (b-a)*rand(1,100)+a (for random.rand values )
                x[i,:] = x[i,:] + (r3 * x[i,:])             # rotation movement update process by Eq. (10.10) 
            logging.info('Init sol at I = {0}: {1}'.format(i, x))
        # bound check
        logging.info('Bound Check')
        for mi in range(N):
            for mj in range(D):
                if x[mi, mj] < x_min[mi, mj]:
                    x[mi, mj] = x_min[mi, mj]
                elif x[mi, mj] > x_max[mi, mj]:
                    x[mi, mj] = x_max[mi, mj]
        logging.info('Init sol after bound check: {0}'.format(x))

        # f = np.apply_along_axis(objfun, 0, x)
        # f = np.amax(x, 0)
        logging.info('Calling obj_fun')
        f = obj_fun(x, Feat, Tr, Tar)
        logging.info('F value returned by obj_fun'.format(f))

        minf = np.amin(f)
        iminf = np.where(f == minf)[0][0].item()
        logging.info('minf and iminf are derived from f returned by obj_fun')
        logging.info('minf: {0}\niminf: {1}'.format(minf, iminf))

        logging.info('best_sub shape: {0}\ngbest shape:{1}'.format(best_sub.shape, gbest.shape))
        logging.info('best_sub value: {0}\ngbest value:{1}'.format(best_sub, gbest))

        if minf <= fgbest:
            logging.info('minf less than fgbest. Update fgbest to minf. Update gbest and best_sub[k,:] to x[minf,:]')
            fgbest = minf
            gbest = x[iminf,:]
            best_sub[k,:] = x[iminf,:]
            fbst[k] = minf
        else:
            logging.info('minf > fgbest. Maintain fbst[k] as fgbest and best_sub[k,:] to gbest')
            fbst[k] = fgbest
            best_sub[k,:] = gbest
        logging.info('Updated fgbest{0}\ngbest: {1}\nbest_sub: {2}\nfbst: {3}'.format(fgbest, gbest, best_sub, fbst))
        
        inewpb = np.where(f <= fpbest)
        pbest[inewpb,:] = x[inewpb,:]
        fpbest[inewpb] = f[inewpb]
        logging.info('Updated pbest: {0}\nUpdated fpbest: {1}'.format(pbest, fpbest))

    bestfit = fbst[-1]
    toc = time.time()
    logging.info('Exiting SSO function')

    return bestfit, fbst, best_sub, toc-tic

# a = np.array([[0.7803, 0.1320, 0.2348, 0.1690, 0.5470], 
#              [0.3897, 0.9421, 0.3532, 0.6491, 0.2963], 
#              [0.2417, 0.9561, 0.8212, 0.7317, 0.7447], 
#              [0.4039, 0.5752, 0.0154, 0.6477, 0.1890], 
#              [0.0965, 0.0598, 0.0430, 0.4509, 0.6868]])

# x_min = np.array([[0.1835, 0.9294, 0.3063, 0.6443, 0.93], 
#                   [0.3685, 0.7757, 0.5085, 0.3786, 0.87], 
#                   [0.6256, 0.4868, 0.5108, 0.8116, 0.55], 
#                   [0.7802, 0.4359, 0.8176, 0.5328, 0.62], 
#                   [0.0811, 0.4468, 0.7948, 0.3507, 0.587]])

# x_max = np.array([[0.2077, 0.1948, 0.3111, 0.9797, 0.59], 
#                   [0.3012, 0.2259, 0.9234, 0.4389, 0.26], 
#                   [0.4709, 0.1707, 0.4302, 0.1111, 0.60], 
#                   [0.2305, 0.2277, 0.1848, 0.2581, 0.71], 
#                   [0.8443, 0.4357, 0.9049, 0.4087, 0.221]])

# print(SSO(a, objfun, x_min, x_max, 10))