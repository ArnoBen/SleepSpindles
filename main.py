# Librairies
import h5py
import numpy as np 
import scipy as sp
import scipy.signal as sg
import matplotlib.pyplot as plt

# Importation des données
data_spindles = h5py.File('data_spindles.h5')    
n_days = len(data_spindles) # n=7
eeg_signals = [None] * 7
hypnograms = [None] * 7
for i in range(n_days):
    path = 'day_' + str(i) + '/'
    eeg_signals[i] = data_spindles[path + 'eeg_signal']
    hypnograms[i] = data_spindles[path + 'hypnogram']
    
#%% Creation d'epochs de 0.5s (duree minimale d'un spindle)
"""A 250Hz, 0.5s <=> 125 points
Tout d'abord, il faut supprimer quelques points pour rendre le nombre total 
de points divisible par 125"""
Fs = 250
n_pts_epochs = 500
for i in range(n_days):
    reste = len(eeg_signals[i])%n_pts_epochs
    eeg_signals[i] = eeg_signals[i][reste:]
    
#Divisons maintenant le signal en epochs de 125 points:
def signal_to_epochs(eeg_data):
    epochs = []
    # n epochs, 0.5s/epoch --> (n, 125)
    for i in range(n_days):
        n_epochs = int(len(eeg_data[i][:])/n_pts_epochs)
        temp_epoch = np.reshape(eeg_data[i][:], (n_epochs, n_pts_epochs))
        epochs.append(temp_epoch)
        return epochs
# exemple pour acceder a day 2 epoch 10 : epochs[2][10,:]
epochs = signal_to_epochs(eeg_signals)
#%%
# Avec une nouvelle estimation du stade  de sommeil toutes les 30s,
# avec 0.5s par epoch, on a une estimation tous les 60 epochs.
def get_sleep_state(day, epoch):
    periode_estimation = 30/(n_pts_epochs/Fs)
    new_estimation = epoch - epoch%periode_estimation
    return hypnograms[day][new_estimation/periode_estimation]
    #A peaufinner, erreurs dans les valeurs extrêmes
#%%
# Filtrons le signal entre 11Hz et 15Hz:
b,a=sg.butter(5,(11/Fs, 15/Fs),'bandpass')
eeg_signals_filt = [None] * 7
for i in range(n_days):
    eeg_signals_filt[i] = sg.filtfilt(b,a,eeg_signals[i])
epochs_filt = signal_to_epochs(eeg_signals_filt)
#%%
plt.psd(epochs_filt[0][9000,:], Fs=500,)
plt.xlim(xmin=0.4, xmax=20)
plt.figure()
plt.plot(epochs_filt[0][9000,:])