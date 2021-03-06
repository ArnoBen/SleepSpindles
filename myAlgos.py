import numpy as np

def threshold_reached(epoch):
    reached = False
    if np.max(epoch) > 15:
        reached = True
    return reached

def nanFound(epoch):
    return np.isnan(epoch).any() # True if nan is found
    
def keepWavePeaks(peaks, peak_heights):
    # Mon objectif est de garde les cretes decroissantes autour du max en conservant les crêtes 2 par 2 
    # (1 à droite et 1 à gauche du max, puis 2 à dte et 2 à gauche, etc.)
    max_height_index = np.argmax(peak_heights)
    new_peaks = []
    new_peaks.append(peaks[np.argmax(peak_heights)])
    new_peak_heights = []
    new_peak_heights.append(np.max(peak_heights))
    
    limit_reached = False
    i = 0
    while not limit_reached:
        # Si on atteint les crêtes aux extrémités
        if  max_height_index + i + 1 == len(peak_heights) or\
            max_height_index - i == 0:
            limit_reached = True
            continue
        # Sinon, on soustrait l'amplitude des crêtes successives.
        # Tant que cette différence est positive, c'est qu'on décroît.
        # On arrête quand la différence est négative.
        else :
            if peak_heights[max_height_index + i] - peak_heights[max_height_index + i + 1] > 0 and \
               peak_heights[max_height_index - i] - peak_heights[max_height_index - i - 1] > 0:
                new_peak_heights.append(peak_heights[max_height_index + i + 1])
                new_peak_heights.insert(0,peak_heights[max_height_index - i - 1])
                new_peaks.append(peaks[max_height_index + i + 1])
                new_peaks.insert(0,peaks[max_height_index - i - 1])
                i += 1           
            else:
                limit_reached = True 
                continue
    return(new_peaks, new_peak_heights)

def waveLength(peaks):
    length = peaks[len(peaks)-1] - peaks[0]
    length = np.round(length/250,1)
    return length

def isTooLong(peaks):
    is_too_long = False
    if waveLength(peaks) > 2: # Si > 2s
        is_too_long = True
    return is_too_long

def isTooShort(peaks):
    is_too_short = False
    if waveLength(peaks) < 0.5: # Si < 0.5s
        is_too_short = True
    return is_too_short

def isSymmetric(peak_heights):
    # Pour tester la symétrie, je compare les crêtes à droite et à gauche du max.
    # Si leur différence de hauteur est relativement faible, on les considérer symétriques.
    
    is_symmetric = True
    n_iter = int(len(peak_heights)/2)
    for i in range(n_iter):
        epsilon = peak_heights[len(peak_heights) - 1 - i] - peak_heights[i]
        if epsilon > 3:
            is_symmetric = False
    return is_symmetric

def isTooHigh(peak_heights):
    is_too_high = False
    if peak_heights[0] > 5 or peak_heights[len(peak_heights)-1] > 5:
        is_too_high = True
    return is_too_high

def pointsToTime(n_points, Fs = 250):
    h, m, s = [], [], []
    n_points = int(n_points/250) # Resultat en secondes
    h, reste = divmod(n_points, 3600)
    m, s = divmod(reste, 60)
    return (h,m,s)