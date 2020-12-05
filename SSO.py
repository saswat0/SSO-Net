import numpy as np
import time
from obj_fun import obj_fun

def SSO(val, objfun, x_min, x_max, kmax):
    N, D = val.shape
    vel = np.random.rand(N, D)
    
    alphak = np.random.rand()
    betak = np.random.rand()

    fbst = np.random.rand(kmax)
    best_sub = np.random.rand(kmax, N)

    t = 1       # time interval stage of k
    gk = np.random.rand()
    x = val.copy()

    # f = np.apply_along_axis(objfun, 0, x)
    f = np.amax(x, 0)

    # find global best and particle best
    fgbest = np.amin(f)
    igbest = np.where(f == fgbest)[0].item()
    
    gbest = x[igbest,:]
    pbest = x
    fpbest = f
    tic = time.time()

    for k in range(kmax):
        for i in range(N):
            for j in range(D):
                r1 = np.random.rand()
                delf = abs(fgbest - f[i]) / fgbest
                term1 = gk * r1 * delf
                if k == 1:
                    vel[i, j] = term1
                else:
                    term2 = min(abs(term1)+alphak*r1*vel[i, j], abs(betak*vel[i, j]))
                    vel[i, j] = term2
            rand = np.random.rand()
            if rand < 0.5:
                x[i,:] = x[i,:] + vel[i,:] * t          # forward movement update process by Eq. (10.9)
            else:
                r3 = 2 * rand - 1                       # (b-a)*rand(1,100)+a (for random.rand values )
                x[i,:] = x[i,:] + (r3 * x[i,:])             # rotation movement update process by Eq. (10.10) 
            
        # bound check
        for mi in range(N):
            for mj in range(D):
                if x[mi, mj] < x_min[mi, mj]:
                    x[mi, mj] = x_min[mi, mj]
                elif x[mi, mj] > x_max[mi, mj]:
                    x[mi, mj] = x_max[mi, mj]

        # f = np.apply_along_axis(objfun, 0, x)
        f = np.amax(x, 0)
        minf = np.amin(f)
        iminf = np.where(f == minf)[0].item()

        if minf <= fgbest:
            fgbest = minf
            gbest = x[iminf,:]
            best_sub[k,:] = x[iminf,:]
            fbst[k] = minf
        else:
            fbst[k] = fgbest
            best_sub[k,:] = gbest
        
        inewpb = np.where(f <= fpbest)
        pbest[inewpb,:] = x[inewpb,:]
        fpbest[inewpb] = f[inewpb]

    bestfit = fbst[-1]
    toc = time.time()

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