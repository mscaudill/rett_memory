import numpy as np

def zero_nans(signal):
    """A nan handler that zeros the values containing nans in signal.

    Args:
        signal (ndarray):       1 x N signal array
    """
    
    signal[np.isnan(signal)] = 0
    return signal

def nanlen(seq):
    """Returns the len of sequence ignoring nans."""

    arr = np.array(seq)
    return np.count_nonzero(~np.isnan(arr))

def crossings(signal, level=0, interpolate=False, nan_handler=zero_nans):
    """Returns nearest or interpolated index locations where signal 
    values cross a constant level.

    signal (seq-like):      1-D signal values whose crossings with level 
                            are to be determined
    level (float):          height at which crossings are to be estimated
                            (Default=0, finds crossings with x-axis)
    interpolate (bool):     whether to estimate crossing locations as floats
                            between the integer indexes of signal using
                            linear interpolation (Default False returns 
                            closest index location to crossing)
    
    Returns: N x 2 array containing paired (opposite signed) crossings.

    Note: For signal values that are tangent to the level, crossings
    returns only the first index or last index of tangency points depending
    on the sign of the derivative at tangency.
    """

    data = nan_handler(np.array(signal))
    height = level
    
    if level < 0:
        data = -1 * data
        height = -1 * height

    #compute the derivative of the sign of the level adjusted data
    derivative = np.diff(np.sign(data-height))
    #obtain all crossings (i.e. where derivative is non-zero)
    crosses = np.nonzero(derivative)[0]
    #prune the crosses
    for idx, (cross, next_cross) in enumerate(zip(crosses, crosses[1:])):
        #get the signs of the derivatives (velocity direction)
        diff_sign = np.sign(derivative[cross])
        next_diff_sign = np.sign(derivative[next_cross])
        #Handle tangency cases
        if (diff_sign == -1 * next_diff_sign ):
            #get data between the crosses and cnt the number
            between = data[cross+1:next_cross+1]
            cnt = len(between)
            #if single point tancency
            if cnt == 1 and  between == height:
                #remove single point tangency (set to -1 for filtering)
                crosses[idx] = crosses[idx+1] = -1
            #else multipoint tangency
            elif cnt > 1 and not (between-height).any() and diff_sign > 0:
                crosses[idx] = cross + 1


        elif diff_sign == next_diff_sign:
            #data value is exactly at threshold and is counted twice
            #i.e. signs go from -1 to 0 then 0 to +1
            if next_cross - cross == 1:
                #zero cross and keep next_cross
                crosses[idx] = -1
            #if cross and next cross are not sequential we have a ledge
            elif derivative[cross] > 0:
                #ledge occurs as signal increases
                #zero next_cross keeping cross
                crosses[idx+1] = -1
            elif derivative[cross] < 1:
                #ledge occurs as signal is decreasing
                #zero this cross keeping next_cross
                crosses[idx] = -1
    #remove zeroed out crossings
    crosses = crosses[crosses > -1]
    #return crosses if none
    if len(crosses) < 1:
        return crosses
    #Signal edges
    if derivative[crosses[0]] < 0:
        #signal is decreasing from start so add NaN as first + cross index
        #prepend = [np.NaN]
        prepend = [0]
    else:
        prepend = []
    if derivative[crosses[-1]] > 0:
        #signal is increasing at end so add NaN as last - cross index
        #postpend = [np.NaN]
        postpend = [len(signal)-1]
    else:
        postpend = []
    #concatenate making a float array of indices
    crosses = np.concatenate([prepend,crosses, postpend]) 
    #Perform interpolation if requested
    if interpolate:
        for idx, cross in enumerate(crosses):
            #ignore nans for interpolation
            if np.isnan(cross):
                continue
            else:
                cross = int(cross)
            if data[cross] == height:
                #if cross is at a datapt no interp needed
                continue 
            else:
                #linear interp
                y0, y1 = data[cross], data[cross+1]
                estimate = (height-y0)/(y1-y0) + cross
                crosses[idx] = estimate
    return crosses

if __name__ == '__main__':

    #test_signal = np.load('signal.npy')
    np.random.seed(2)
    test_signal = 0.3*np.random.random(1000) - \
                  np.sin(np.linspace(0, 2*np.pi, 1000))
    
    crosses = crossings(test_signal, level=1, interpolate=True)

