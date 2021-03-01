import os
import librosa
import numpy as np
from python_speech_features import mfcc
# from Normalize 
import normalize
from sklearn import preprocessing
import scipy
from spafe.features.gfcc import gfcc
from random import shuffle, uniform
import scipy.io.wavfile as wav
from sklearn.decomposition import PCA
from scipy import signal
import scipy.signal
# from Prep_feat import feature
import random as rn
# from GWO import GWO
# from PSO import PSO
from SSO import SSO
# from GSO import GSO
# from Proposed import Proposed
from net.CNN_model import CNN_model
# from plot_results import plot_res
# import getGFCC
from spafe.features.gfcc import gfcc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from encoder import LabelEncoder
import speechpy
import warnings
import numpy.matlib
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(filename='SSO.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Program Start')

print('Processing...')
ch = int(input("Create Features: 1-yes, 0-no"))
encoder = LabelEncoder()
mfcc_cens_vectors = []
label_vectors = []
outputs = np.zeros(shape=(50))
idx = 0
if ch == 1:
    n = 0
    filename = 'vox1_dev_wav/'
    info = os.listdir(filename)
    for i in range(len(info)):  # range(num):    # Get the files from each id folder
        
        path = filename + info[i]    # Path of the info folder
        print(path)
        
        info1 = os.listdir(path)
        for j in range(len(info1)):  #Get the files in the folder inside each id
            #print(j)
            path1 = path + '/' + info1[j]  # Path of the info1 folder
            info2 = os.listdir(path1)  # Get the audio files

            for k in range(len(info2)):  # For each audio files
                path2 = path1 + '/' + info2[k]
                sig, rate = librosa.load(path2, duration=3.0)

                #rate, sig = wav.read(path2)   # Read .wav signal
                sig = scipy.signal.medfilt(sig , kernel_size=None)  # Filtering
                mfcc1 = librosa.feature.mfcc(sig, rate, n_mfcc = 13) #, hop_length=800, n_fft=1600)
                #print(mfcc1.shape)
                librosa_stft = np.abs(librosa.stft(sig))

                cent = librosa.feature.spectral_centroid(y=sig, sr=rate)
                #print(cent.shape)
                flat = librosa.feature.spectral_flatness(y=sig)
                #print(flat.shape)
                rolloff = librosa.feature.spectral_rolloff(y=sig, sr=rate, roll_percent=0.99)
                #print(rolloff.shape)
                zcr = librosa.feature.zero_crossing_rate(sig)
                #print(zcr.shape)


                Tot_feat = np.vstack((mfcc1, librosa_stft, cent, flat, rolloff, zcr))
                #Tot_feat = speechpy.processing.cmvn(Tot_feat,variance_normalization=True)
                #print(Tot_feat.shape)
                Tot_feat = Tot_feat.T
                Tot_feat = np.mean(Tot_feat,axis=0)
                #print(Tot_feat.shape)
                Tot_feat =Tot_feat.reshape(1,-1)
                #print(Tot_feat.shape)
                label = info[i]
                print(label)
                label_id = encoder.add(label)
                label_vectors.append(label_id)
                labels = np.zeros((1, 20))
                #labels = np.zeros((1, num))
                #print(labels)
                lab = int(path[len(path)-4:len(path)])
                print(lab)
                labels[0,lab-1] = 1
                

                #outputs[idx] = label
                #idx = idx + 1
                if n == 0:
                   Feat = Tot_feat #[0,:]
                   Tar = labels
                   Tr = lab
                   n = 1

                   
                else:
                    Feat = np.concatenate((Feat,Tot_feat)) #,axis = 0)
                    #print('Feat.shape')
                    #print(Feat.shape)
                    Tar = np.concatenate((Tar,labels),axis = 0)
                    Tr = np.concatenate((Tr, lab), axis=None)
   

    print(Feat.shape)   # (2787, 1042)
    print(Tar.shape)    # (2787, 20)
    print(Tr.shape)     # (2787,)
    print(label_vectors)
    #results = CNN_model(Feat, Tar, Tr, 0.85*100, 200, 2)
    #np.save('tempResults.npy',results)
    np.save('Feat_020121.npy', Feat)
    np.savetxt("Feat_020121.csv", Feat, delimiter=",")
    np.save('Labels_020121.npy', Tar)
    np.savetxt("Labels_020121.csv", Tar, delimiter=",")
    np.save('Labels_encode_020121.npy', label_vectors)
    np.savetxt("Labels_encode_020121.csv", label_vectors, delimiter=",")

else:
    Feat = np.load('Feat_020121.npy')
    Tar = np.load('Labels_020121.npy')
    Tr = np.load('Labels_encode_020121.npy')
    print(Feat.shape)
    print(Tar.shape)
    print(Tr.shape)
    logging.info('Feat shape: {0}\nTar shape: {1}\nTr shape: {2}'.format(Feat.shape, Tar.shape, Tr.shape))
    # results = CNN_model(Feat, Tar, Tr, 0.85*100, 200, 2)
    # np.save('100SprsResults.npy',results)
    


####################### Optimal feature selection and Classification#########################
an = 1
if an == 1:
    sequence = [v2 for v2 in range(Feat.shape[0])]
    shuffle(sequence)  # Shuffled Index
    Feat = Feat[sequence, :]
    Tar = Tar[sequence, :]
    Tr = Tr[sequence]
    logging.info('Shuffled Feat, Tar, Tr')
    logging.info('New values for Feat: {0}\nTar: {1}\nTr: {2}'.format(Feat, Tar, Tr))

    Npop = 20  # population size
    ch_len = 18 + 1  # Solution length
    xmin = np.matlib.repmat(np.concatenate([np.zeros((1, ch_len)), 5], axis=None), Npop, 1)
    xmax = np.matlib.repmat(np.concatenate([Feat.shape[1]-1 * np.ones((1, ch_len)), 255], axis=None), Npop, 1)
    initsol = np.zeros((Npop, xmax.shape[1]))
    logging.info('Npop: {0}\nCh_len: {1}\nxmin shape: {2}\nxmax shape: {3}\ninit sol shape: {4}'.format(Npop, ch_len, xmin.shape, xmax.shape, initsol.shape))
    for p1 in range(Npop):
        for p2 in range(xmax.shape[1]):
            initsol[p1, p2] = uniform(xmin[p1, p2], xmax[p1, p2])
    fname = 'Obj_fun'
    Max_iter = 1
    # print(initsol.shape)
    logging.info('Max_iter: {0}\nxmin: {1}\nxmax: {2}\ninit sol: {3}'.format(Max_iter, xmin, xmax, initsol))

    logging.info('SSO Function Call')
    [bestfit, fitness, bestsol, time] = SSO(initsol, fname, xmin, xmax, Max_iter, Feat, Tar, Tr)
    logging.info('Bestfit: {0}\nFitness: {1}\nBestsol: {2}\nTime: {3}'.format(bestfit, fitness, bestsol, time))

    best_solutions = np.unique(np.round(bestsol))
    logging.info('Best solution: {0}'.format(best_solutions))

    np.save('bestsol.npy', bestsol)
    

    # array([[ 724.19058656,  925.23949655,  983.81622341,  489.02540102,
    #      248.30276763,  247.83312184,  182.64376998,  499.81998313,
    #       17.24889977,  855.42830708,  194.96897484,  296.60791002,
    #      227.41057205, 1013.77545639,  603.99362592,  760.296382  ,
    #      743.10882099,  233.41152031,  396.72476572,  192.31126365]])

    # array([[ 918.22556253,  392.73049752,  863.17733115,  513.94602543,
    #      827.09969782,  304.92198609, 1015.6800973 ,  342.66882845,
    #      266.612029  ,  209.93503492,  764.2901169 ,   52.48465795,
    #      665.01243928,  122.10629256,  665.52894985,  492.46032004,
    #      239.90272613,  211.05901204, 1004.13359065,  251.79266279]])

    # array([[  25.7714522 ,  250.41511423, 1041.        ,  808.5544641 ,
    #     1041.        ,  457.98179344,  593.55701581, 1041.        ,
    #     1041.        ,  510.6197602 ,  849.47680896, 1041.        ,
    #     1041.        ,  474.71248455, 1041.        , 1041.        ,
    #     1041.        , 1041.        , 1041.        ,  231.35259898],
    #    [ 500.05504768, 1041.        , 1041.        , 1041.        ,
    #     1041.        , 1041.        , 1041.        , 1041.        ,
    #     1041.        ,  956.35357251, 1041.        , 1041.        ,
    #      962.4752075 , 1041.        , 1041.        , 1041.        ,
    #     1041.        ,  517.9198227 , 1041.        ,  255.        ]])