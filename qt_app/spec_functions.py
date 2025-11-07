"""
PySpectrometer2 Functions - Adapted for Qt Integration
Les Wright 2022 - https://github.com/leswright1977/PySpectrometer

Core spectral processing functions including:
- Savitzky-Golay filtering
- Peak detection
- Graticule generation
"""

import numpy as np


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Apply Savitzky-Golay filter for smoothing."""
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int32(window_size))
        order = np.abs(np.int32(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    
    # Precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    
    # Pad the signal at the extremes with values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    
    return np.convolve(m[::-1], y, mode='valid')


def peakIndexes(y, thres=0.3, min_dist=1, thres_abs=False):
    """
    Peak detection using threshold and minimum distance.
    From peakutils library.
    """
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    if not thres_abs:
        thres = thres * (np.max(y) - np.min(y)) + np.min(y)

    min_dist = int(min_dist)

    # Compute first order difference
    dy = np.diff(y)

    # Propagate left and right values successively to fill all plateau pixels (0-value)
    zeros, = np.where(dy == 0)

    # Check if the signal is totally flat
    if len(zeros) == len(y) - 1:
        return np.array([])

    if len(zeros):
        # Compute first order difference of zero indexes
        zeros_diff = np.diff(zeros)
        # Check when zeros are not chained together
        zeros_diff_not_one, = np.add(np.where(zeros_diff != 1), 1)
        # Make an array of the chained zero indexes
        zero_plateaus = np.split(zeros, zeros_diff_not_one)

        # Fix if leftmost value in dy is zero
        if zero_plateaus[0][0] == 0:
            dy[zero_plateaus[0]] = dy[zero_plateaus[0][-1] + 1]
            zero_plateaus.pop(0)

        # Fix if rightmost value of dy is zero
        if len(zero_plateaus) and zero_plateaus[-1][-1] == len(dy) - 1:
            dy[zero_plateaus[-1]] = dy[zero_plateaus[-1][0] - 1]
            zero_plateaus.pop(-1)

        # For each chain of zero indexes
        for plateau in zero_plateaus:
            median = np.median(plateau)
            # Set leftmost values to leftmost non zero values
            dy[plateau[plateau < median]] = dy[plateau[0] - 1]
            # Set rightmost and middle values to rightmost non zero values
            dy[plateau[plateau >= median]] = dy[plateau[-1] + 1]

    # Find the peaks by using the first order difference
    peaks = np.where(
        (np.hstack([dy, 0.0]) < 0.0)
        & (np.hstack([0.0, dy]) > 0.0)
        & (np.greater(y, thres))
    )[0]

    # Handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks


def generateGraticule(wavelengthData):
    """
    Generate graticule positions for spectrum display.
    Returns: [tens_positions, fifties_data]
    """
    low = wavelengthData[0]
    high = wavelengthData[len(wavelengthData) - 1]
    
    low = int(round(low)) - 10
    high = int(round(high)) + 10
    
    returndata = []
    
    # Find positions of every whole 10nm
    tens = []
    for i in range(low, high):
        if (i % 10 == 0):
            position = min(enumerate(wavelengthData), key=lambda x: abs(i - x[1]))
            if abs(i - position[1]) < 1:
                tens.append(position[0])
    returndata.append(tens)
    
    # Vertical lines every whole 50nm
    fifties = []
    for i in range(low, high):
        if (i % 50 == 0):
            position = min(enumerate(wavelengthData), key=lambda x: abs(i - x[1]))
            if abs(i - position[1]) < 1:
                labelpos = position[0]
                labeltxt = int(round(position[1]))
                labeldata = [labelpos, labeltxt]
                fifties.append(labeldata)
    returndata.append(fifties)
    
    return returndata
