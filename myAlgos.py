import numpy as np

def threshold_reached(epoch):
    reached = False
    if np.max(epoch) > 18:
        reached = True
    return reached

def nanFound(epoch):
    return np.isnan(epoch).any() #True if nan is found
    
def keepWavePeaks(peaks, peak_heights):
    #Garde les cretes decroissantes autour du max
    max_height_index = np.argmax(peak_heights)
    print(max_height_index)
    
    