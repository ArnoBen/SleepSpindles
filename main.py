# Librairies
import numpy as np 
import scipy.signal as sg
import matplotlib.pyplot as plt
import h5py
import scipy
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
    
# Filtrons le signal entre 11Hz et 15Hz:
Fs = 250
b,a=sg.butter(5,(11/Fs, 15/Fs),'bandpass')
eeg_signals_filt = [None] * n_days
for i in range(n_days):
    eeg_signals_filt[i] = sg.filtfilt(b,a,eeg_signals[i])
# Repartition du signal en epochs

n_pts_epochs = 1250
# Il faut rendre le nombre de pts du signal divisible par la taille d'un epoch
for i in range(n_days):
    reste = len(eeg_signals[i])%n_pts_epochs
    eeg_signals[i] = eeg_signals[i][reste:]
    eeg_signals_filt[i] = eeg_signals_filt[i][reste:]
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

epochs = signal_to_epochs(eeg_signals)
epochs_filt = signal_to_epochs(eeg_signals_filt)


#%% Nettoyage du signal
importlib.reload(myAlgos)
eeg_signals_filt_nan = [None] * n_days
for i in range(7):
    #Enlever les valeurs excessivement grandes
    amp_excess = np.where(np.abs(eeg_signals_filt[i]) > 80)[0]
    eeg_signals_filt_nan[i] = np.copy(eeg_signals_filt[i])
    rem_len = 375 #On enlève 1.5s à droite et à gauche
    for j in range(len(amp_excess)):
        #Si on est pas aux limites
        if amp_excess[j] > rem_len and amp_excess[j] + rem_len < len(eeg_signals_filt_nan[i]):
            eeg_signals_filt_nan[i][amp_excess[j] - rem_len :  amp_excess[j] + rem_len] = np.NaN
        #Sinon
        elif amp_excess[j] < rem_len:
            eeg_signals_filt_nan[i][:amp_excess[j] + rem_len] = np.NaN
        elif amp_excess[j] + rem_len > len(eeg_signals_filt_nan[i]):
            eeg_signals_filt_nan[i][amp_excess[j]:] = np.NaN
    
epochs_filt_nan = signal_to_epochs(eeg_signals_filt_nan)

#Detection des spindles
spindles_detected = 0
spindles_positions = [None] * n_days
spindles_heights = [None] * n_days
spindles_length = [None] * n_days
fails = np.array([0,0,0,0,0,0])
for i in range(n_days): #Parcours par jour
    spindles_positions[i] = []
    spindles_heights[i] = []
    spindles_length[i] = []
    for j in range(epochs_filt_nan[i].shape[0]): #Parcours par epoch
        if j == 0: continue #petit probleme au tout debut a cause d'artefacts
        current_epoch = epochs_filt_nan[i][j]
        if (not myAlgos.nanFound(current_epoch)) and \
            myAlgos.threshold_reached(current_epoch) :
            
            # On se centre autour de la valeur max.
            # 600 points <=> 2.4 secondes
            max_pos = np.argmax(current_epoch)
            max_val = np.max(current_epoch)
            window = eeg_signals_filt_nan[i][j*n_pts_epochs + max_pos - 300 : j*n_pts_epochs + max_pos + 300]
            #C'est plus pratique d'avoir le spindle centré sur sa valeur absolue max:
            if window[int(len(window)/2)] < np.max(-window[int(len(window)/2) - 20 : int(len(window)/2) + 20]):
                window = -window
            if not myAlgos.nanFound(window):    
                peaks, peak_properties = sg.find_peaks(window, height=0)
                peak_heights = peak_properties['peak_heights']
                if max_val == np.max(peak_heights) : #si rien de bizare lors du fenetrage
                    wave_peaks, wave_heights = myAlgos.keepWavePeaks(peaks, peak_heights)
                    if not myAlgos.isTooShort(wave_peaks) and not myAlgos.isTooLong(wave_peaks): #si 0.5<x<2
                        if myAlgos.isSymmetric(wave_heights): #si les cretes sont symmetriques
                            if myAlgos.isTooHigh(wave_heights): #Si l'amplitude min n'est pas trop grande
                                spindles_detected += 1
                                spindles_positions[i].append(j*n_pts_epochs + max_pos)
                                spindles_heights[i].append(np.round(np.max(wave_heights),0))
                                spindles_length[i].append(myAlgos.waveLength(wave_peaks))
                            else :
                                fails[5] += 1
                                continue
                        else :
                            fails[4] += 1
                            continue
                    else :
                        fails[3] += 1
                        continue
                else : 
                    fails[2] += 1
                    continue
            else :
                fails[1] += 1
                continue
        else : 
            fails[0] += 1
            continue 
print('spindles detectees : ', spindles_detected)           

#%% Afficher une spindle
plt.clf()
myspindle = 132
plt.plot(eeg_signals_filt_nan[3][spindles_positions[3][myspindle] - 2500 : spindles_positions[3][myspindle] + 2500])
#plt.axhline(15,color='C1', alpha = 0.8)
#%% Affichage de spindles isolees
j = 1
for i in range(20):
    rand_day = int(np.random.randint(0,7,1))
    rand_spindle_pos = spindles_positions[rand_day][int(np.random.randint(0,200,1))]
    plt.subplot(1,4,j)
    plt.plot(eeg_signals_filt_nan[rand_day][rand_spindle_pos - 250 : rand_spindle_pos + 250])
    plt.title(str(rand_day) + ' | ' + str(rand_spindle_pos))
    j += 1
    if j == 5 : 
        plt.figure()
        j = 1
#%% Affichage des spindles sur tout le signal d'un jour
jour = 3
plt.figure()
plt.plot(eeg_signals_filt_nan[jour])
plt.ylim(-500,500)
for i in range(len(spindles_positions[jour])):
    plt.axvline(x = spindles_positions[jour][i], ymin = 0.4, ymax = 0.6, color = 'r')
plt.plot(hypnograms_long[jour] * 100, color = 'C1')    

#%% Repartition des epochs par stade de sommeil:
stage = [0] * 5
for i in range(7):
    for pos in spindles_positions[i]:
        stage[int(hypnograms_long[i][pos])] += 1
#plt.clf()
#
#fig = plt.figure(figsize = (10,5))
## Histogramme par stade de sommeil
#plt.barh(np.arange(5), stage)
#plt.xlabel("nb d'occurrences")
#plt.yticks(np.arange(5),('Eveil', 'Sommeil léger N1', 'Sommeil léger N2', 'Sommeil profond N3', 'REM'))
#
#plt.show()  
#%% Densité des spindles
parts = 2 # Je sépare arbitrairement une nuit en 10 parties égales
spindle_density = np.zeros([n_days, parts])
spindle_density_mean = np.zeros(parts)
spindle_density_std = np.zeros(parts)

plt.figure()
for i in range(n_days):
    l = len(eeg_signals[i])
    for j in range(parts):
        temp = np.array(spindles_positions[i])
        bool_table = np.equal(temp < j*l/parts, temp > (j+1)*l/parts)
        spindle_density[i,j] = (sum(bool_table))
    plt.plot(spindle_density[i])

for k in range(parts): 
    spindle_density_mean[k] = np.mean(spindle_density[:,k])
    spindle_density_std[k] = np.std(spindle_density[:,k])
plt.figure()
plt.errorbar(np.arange(parts), spindle_density_mean, spindle_density_std, linestyle='-', marker='o', capsize=5)

# T-test
t, p = scipy.stats.ttest_ind(spindle_density[:,0], spindle_density[:,1])
print(t)

#%%
duration_S2_tot = np.sum(np.concatenate(hypnograms_long) == 2)
time_tot = myAlgos.pointsToTime(duration_S2_tot)
min_tot = time_tot[0] * time_tot[1]
    
stage[2]/min_tot
#%%


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