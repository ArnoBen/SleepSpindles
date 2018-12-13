# Librairies
import numpy as np 
import scipy
import scipy.signal as sg
import matplotlib.pyplot as plt
import h5py

# Importation des donnees
data_spindles = h5py.File('data_spindles.h5')    
n_days = len(data_spindles) # n=7
eeg_signals = [None] * n_days
hypnograms = [None] * n_days
hypnograms_long = [None] * n_days

for i in range(n_days):
    path = 'day_' + str(i) + '/'
    eeg_signals[i] = data_spindles[path + 'eeg_signal']
    hypnograms[i] = data_spindles[path + 'hypnogram']

# Cela permet de recuperer un stade de sommeil facilement :
for i in range(n_days):
    hypnograms_long[i] = np.empty(len(eeg_signals[i]))
    hypnograms_long[i][:] = np.nan
    len_cast = int(len(eeg_signals[i])/len(hypnograms[i]))
    for j in range(len(hypnograms[i])):
        hypnograms_long[i][j*len_cast:(j+1)*len_cast] = hypnograms[i][j]
    hypnograms_long[i][np.where(np.isnan(hypnograms_long[i]))] = 0
# Ainsi, stade de sommeil du point eeg_signals[i][124] : hypnograms_long[124] 
#%% Creation d'epochs de 0.5s (duree minimale d'un spindle)
"""A 250Hz, 0.5s <=> 125 points
Tout d'abord, il faut supprimer quelques points pour rendre le nombre total 
de points divisible par 125"""
Fs = 250
n_pts_epochs = 125
for i in range(n_days):
    reste = len(eeg_signals[i])%n_pts_epochs
    eeg_signals[i] = eeg_signals[i][reste:]
    hypnograms_long[i] = hypnograms_long[i][reste:]
    
#Divisons maintenant le signal en epochs de 125 points:
def signal_to_epochs(eeg_data):
    epochs = [None] * n_days
    # n epochs, 0.5s/epoch --> (n, 125)
    for i in range(n_days):
        n_epochs = int(len(eeg_data[i])/n_pts_epochs)
        temp_epoch = np.reshape(eeg_data[i], (n_epochs, n_pts_epochs))
        epochs[i] = temp_epoch
    return epochs
#for i in range(n_days):
epochs = signal_to_epochs(eeg_signals)
# exemple pour acceder a day 2 epoch 10 : epochs[2][10,:]
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
eeg_signals_filt = [None] * n_days
for i in range(n_days):
    eeg_signals_filt[i] = sg.filtfilt(b,a,eeg_signals[i])
epochs_filt = signal_to_epochs(eeg_signals_filt)
#%% Repartition des epochs par stade de sommeil:
hypno_epochs = signal_to_epochs(hypnograms_long)
epochs_stade = [None] * n_days
for i in range(7) : 
    epochs_stade[i] = [None] * 5 
    for j in range(5):
        epochs_stade[i][j] = epochs_filt[0][np.where(hypno_epochs[0][:,0]==j)]
#%%
# Calcul psd pour day 0
psd, moy, std = [[None] * 5 for i in range(3)]
for i in range(1,5): #on exclut l'eveil
    
    psd[i] = sg.periodogram(epochs_stade[0][i],fs=Fs)[1][:,11:16]
    # Enlever les valeurs aberrantes
    amp_excess = np.where(np.mean(psd[i],axis=1)>20)
    zero_values = np.where(np.mean(psd[i],axis=1)==0.0)
    psd[i][amp_excess] = np.NaN
    moy[i] = np.nanmean(psd[i])
    std[i] = np.nanstd(psd[i])
plt.errorbar([1,2,3,4],moy[1:5],std[1:5],linestyle='None', marker='o')
#PENSER A UTILISER LA P VALUE

#%%
#plt.psd(epochs_filt[0][9000,:], Fs=500,)
#plt.xlim(xmin=0.4, xmax=20)
#plt.figure()
#plt.plot(epochs_filt[0][9000,:])

""" Idees pour le filtre :
    - Std > seuil
    - si une ou plusieurs valeurs = 0 precisement, eeg deco
    
"""