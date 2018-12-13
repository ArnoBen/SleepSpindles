# Librairies
import numpy as np 
import scipy
import scipy.signal as sg
import matplotlib.pyplot as plt
import h5py
import importlib
import myAlgos

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
#%% Repartition du signal en epochs
Fs = 250
n_pts_epochs = 2500
# Il faut rendre le nombre de pts du signal divisible par la taille d'un epoch
for i in range(n_days):
    reste = len(eeg_signals[i])%n_pts_epochs
    eeg_signals[i] = eeg_signals[i][reste:]
    hypnograms_long[i] = hypnograms_long[i][reste:]
    
def signal_to_epochs(eeg_data):
    epochs = [None] * n_days
    # n epochs, 0.5s/epoch --> (n, 125)
    for i in range(n_days):
        n_epochs = int(len(eeg_data[i])/n_pts_epochs)
        temp_epoch = np.reshape(eeg_data[i], (n_epochs, n_pts_epochs))
        epochs[i] = temp_epoch
    return epochs
# exemple pour acceder a day 2 epoch 10 : epochs[2][10,:]

# Filtrons le signal entre 11Hz et 15Hz:
b,a=sg.butter(5,(11/Fs, 15/Fs),'bandpass')
eeg_signals_filt = [None] * n_days
for i in range(n_days):
    eeg_signals_filt[i] = sg.filtfilt(b,a,eeg_signals[i])
  
epochs = signal_to_epochs(eeg_signals)
epochs_filt = signal_to_epochs(eeg_signals_filt)

#%% Nettoyage du signal
eeg_signals_filt_nan = [None] * n_days
for i in range(7):
    #Enlever les valeurs excessivement grandes
    amp_excess = np.where(np.abs(eeg_signals_filt[i]) > 50)[0]
    eeg_signals_filt_nan[i] = np.copy(eeg_signals_filt[i])
    eeg_signals_filt_nan[i][amp_excess] = np.NaN

epochs_filt_nan = signal_to_epochs(eeg_signals_filt_nan)

#Detection des spindles

for i in range(n_days): #Parcours par jour
    for j in range(epochs_filt_nan[i].shape[0]): #Parcours par epoch
        current_epoch = epochs_filt_nan[i][j]
        if (not myAlgos.nanFound(current_epoch)) and \
            myAlgos.threshold_reached(current_epoch) :
            
            # On se centre autour de la valeur max.
            # 750 points <=> 3 secondes
            max_pos = np.argmax(current_epoch)
            window = eeg_signals_filt_nan[i][j*n_pts_epochs + max_pos - 375 : j*n_pts_epochs + max_pos + 375]
            if not myAlgos.nanFound(window):    
                peaks, peak_properties = sg.find_peaks(window, height=0)
                peak_heights = peak_properties['peak_heights']
                myAlgos.keepWavePeaks(peaks, peak_heights)
            else :
                continue
        else :
            continue
               
           
        
#%% Repartition des epochs par stade de sommeil:
hypno_epochs = signal_to_epochs(hypnograms_long)
def epoch_to_sleepstage(epochs):
    epochs_stade = [None] * n_days
    for i in range(7) : 
        epochs_stade[i] = [None] * 5 
        for j in range(5):
            epochs_stade[i][j] = epochs[0][np.where(hypno_epochs[0][:,0]==j)]
    return epochs_stade
epochs_stade_filt = epoch_to_sleepstage(epochs_filt)
#%% Detection des spindles
eeg_signals_filt_nan = [None] * n_days
for i in range(7):
    amp_excess = np.where(np.abs(eeg_signals_filt[i]) > 50)
    eeg_signals_filt_nan[i] = np.copy(eeg_signals_filt[i])
    eeg_signals_filt_nan[i][amp_excess] = np.NaN

epochs_filt_nan = signal_to_epochs(eeg_signals_filt_nan)
epochs_stade_filt_nan = epoch_to_sleepstage(epochs_filt_nan)

"""
On estime tout d'abord la moyenne et l'ecart-type du signal par phase.
Ensuite, on regarde combien de spindles sont détectés (par minute, accessoirement) 
"""

# Essai sur day 0, phase 2
std_test = np.nanstd(epochs_stade_filt_nan[0][2])
detected_epochs = np.where(np.nanmax(epochs_stade_filt_nan[0][2],axis=1) > 3 * std_test)[0]

temp_detected = []
len_det = len(detected_epochs)
for i in range(len_det):
    # S'il y a une valeur Nan, next
    if np.isnan(epochs_stade_filt_nan[0][2][detected_epochs[i],:]).any() == True:
        continue
    if  (i < len_det - 2 and detected_epochs[i + 2] == detected_epochs[i] + 2) or \
        (i > 2 and detected_epochs[i - 2] == detected_epochs[i] - 2):
        continue
        # + 2 parce que si un spindle se passe sur 2 epochs, il faut la garder
        # mais 3 epochs c'est trop long.
    else:  
        temp_detected.append(detected_epochs[i])
detected_epochs = temp_detected

epochs_stade_filt_nan[0][2][detected_epochs].shape

#Essai detection de cretes
peaks, peak_properties = sg.find_peaks(epochs_stade_filt_nan[0][2][detected_epochs[41]], height=0)

plt.plot(epochs_stade_filt_nan[0][2][detected_epochs[41]])
plt.scatter(peaks,peak_properties['peak_heights'], color = 'r')


#%%
# Calcul psd pour day 0
psd, moy, std = [[None] * 5 for i in range(3)]
for i in range(1,5): #on exclut l'eveil   
    psd[i] = sg.periodogram(epochs_filt_stade[0][i],fs=Fs)[1][:,11:16]
    # Enlever les valeurs epochs_stade_filt
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
#%%
# Avec une nouvelle estimation du stade  de sommeil toutes les 30s,
# avec 0.5s par epoch, on a une estimation tous les 60 epochs.
#def get_sleep_state(day, epoch):
#    periode_estimation = 30/(n_pts_epochs/Fs)
#    new_estimation = epoch - epoch%periode_estimation
#    return hypnograms[day][new_estimation/periode_estimation]
#    #A peaufinner, erreurs dans les valeurs extrêmes