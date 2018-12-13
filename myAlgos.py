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
    
    new_peaks = []
    new_peaks.append(peaks[np.argmax(peak_heights)])
    new_peak_heights = []
    new_peak_heights.append(np.max(peak_heights))
    
    right_limit_reached = False
    i = 0
    while not right_limit_reached:
        #Tant qu'on decroit      
        if max_height_index + i + 1 == len(peak_heights):
            right_limit_reached = True
            continue
        if peak_heights[max_height_index + i] - peak_heights[max_height_index + i + 1] > 0:
            new_peak_heights.append(peak_heights[max_height_index + i + 1])
            new_peaks.append(peaks[max_height_index + i + 1])
            i += 1
        else:
            right_limit_reached = True
    
    left_limit_reached = False
    i = 0
    while not left_limit_reached:
        #Tant qu'on decroit      
        if max_height_index - i == 0:
            left_limit_reached = True
            continue
        if peak_heights[max_height_index - i] - peak_heights[max_height_index - i - 1] > 0:
            new_peak_heights.insert(0,peak_heights[max_height_index - i - 1])
            new_peaks.insert(0,peaks[max_height_index - i - 1])
            i += 1
        else:
            left_limit_reached = True
    
    return(new_peaks, new_peak_heights)
    