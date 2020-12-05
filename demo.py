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
#import getGFCC
from spafe.features.gfcc import gfcc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from encoder import LabelEncoder
import speechpy
import warnings
warnings.filterwarnings("ignore")


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
                labels = np.zeros((1, 1251))
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
    print(Tar.shape)    # (2787, 1251)
    print(Tr.shape)     # (2787,)
    print(label_vectors)
    #results = CNN_model(Feat, Tar, Tr, 0.85*100, 200, 2)
    #np.save('tempResults.npy',results)
    np.save('Feat_291120.npy', Feat)
    np.save('Labels_291120.npy', Tar)
    np.save('Labels_encode_291120.npy', label_vectors)
else:
    Feat = np.load('Feat_291120.npy')
    Tar = np.load('Labels_291120.npy')
    Tr = np.load('Labels_encode_291120.npy')
    print(Feat.shape)
    print(Tar.shape)
    print(Tr.shape)
    results = CNN_model(Feat, Tar, Tr, 0.85*100, 200, 2)
    np.save('100SprsResults.npy',results)
    


####################### Optimal feature selection and Classification#########################
an = 0
if an == 1:
    sequence = [v2 for v2 in range(Feat.shape[0])]
    shuffle(sequence)  # Shuffled Index
    Feat = Feat[sequence, :]
    Tar = Tar[sequence, :]
    Tr = Tr[sequence]

    Npop = 10  # population size
    ch_len = 28 +1  # Solution length
    xmin = np.matlib.repmat(np.concatenate([np.zeros((1, ch_len)), 5], axis=None), Npop, 1)
    xmax = np.matlib.repmat(np.concatenate([Feat.shape[1]-1 * np.ones((1, ch_len)), 255], axis=None), Npop, 1)
    initsol = np.zeros((xmax.shape))
    for p1 in range(Npop):
        for p2 in range(xmax.shape[1]):
            initsol[p1, p2] = uniform(xmin[p1, p2], xmax[p1, p2])
    fname = 'Obj_fun'
    Max_iter = 25

    print("SSO...")
    [bestfit, fitness, bestsol, time] = SSO(initsol, fname, xmin, xmax, Max_iter,Feat, Tar, Tr)  

    np.save('bestsol.npy', bestsol)