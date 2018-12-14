import numpy as np

def threshold_reached(epoch):
    reached = False
    if np.max(epoch) > 14:
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
    
    limit_reached = False
    i = 0
    while not limit_reached:
        if  max_height_index + i + 1 == len(peak_heights) or\
            max_height_index - i == 0:
            limit_reached = True
            #TODO
            continue
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

def isTooLong(peaks):
    is_too_long = False
    if peaks[len(peaks)-1] - peaks[0] > 575: #Si la wave > 2.3s
        is_too_long = True
    return is_too_long

def isTooShort(peaks):
    is_too_short = False
    if peaks[len(peaks)-1] - peaks[0] < 100: #Si la wave < 0.4s
        is_too_short = True
    return is_too_short

def isSymmetric(peak_heights):
    is_symmetric = True
    #Si pair:
    if len(peak_heights)%2 == 0:
        n_iter = int(len(peak_heights)/2)
        for i in range(n_iter):
            epsilon = peak_heights[len(peak_heights) - 1 - i] - peak_heights[i]
            if epsilon > 3:
                is_symmetric = False
                return is_symmetric
    #Si impair:
    if len(peak_heights)%2 == 1:
        n_iter = int(len(peak_heights)/2)
        for i in range(n_iter):
            epsilon = peak_heights[len(peak_heights) - 1 - i] - peak_heights[i]
            if epsilon > 3:
                is_symmetric = False
                return is_symmetric
    return is_symmetric
                