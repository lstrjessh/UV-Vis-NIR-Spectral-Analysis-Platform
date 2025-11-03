"""
PySpectrometer2 Functions - Adapted for Qt Integration
Les Wright 2022 - https://github.com/leswright1977/PySpectrometer

Core spectral processing functions including:
- Wavelength to RGB conversion
- Savitzky-Golay filtering
- Peak detection
- Calibration data management
- Graticule generation
"""

import numpy as np
import time


def wavelength_to_rgb(nm):
    """Convert wavelength (nm) to RGB color tuple."""
    gamma = 0.8
    max_intensity = 255
    factor = 0
    rgb = {"R": 0, "G": 0, "B": 0}
    
    if 380 <= nm <= 439:
        rgb["R"] = -(nm - 440) / (440 - 380)
        rgb["G"] = 0.0
        rgb["B"] = 1.0
    elif 440 <= nm <= 489:
        rgb["R"] = 0.0
        rgb["G"] = (nm - 440) / (490 - 440)
        rgb["B"] = 1.0
    elif 490 <= nm <= 509:
        rgb["R"] = 0.0
        rgb["G"] = 1.0
        rgb["B"] = -(nm - 510) / (510 - 490)
    elif 510 <= nm <= 579:
        rgb["R"] = (nm - 510) / (580 - 510)
        rgb["G"] = 1.0
        rgb["B"] = 0.0
    elif 580 <= nm <= 644:
        rgb["R"] = 1.0
        rgb["G"] = -(nm - 645) / (645 - 580)
        rgb["B"] = 0.0
    elif 645 <= nm <= 780:
        rgb["R"] = 1.0
        rgb["G"] = 0.0
        rgb["B"] = 0.0
    
    if 380 <= nm <= 419:
        factor = 0.3 + 0.7 * (nm - 380) / (420 - 380)
    elif 420 <= nm <= 700:
        factor = 1.0
    elif 701 <= nm <= 780:
        factor = 0.3 + 0.7 * (780 - nm) / (780 - 700)
    
    if rgb["R"] > 0:
        rgb["R"] = int(max_intensity * ((rgb["R"] * factor) ** gamma))
    else:
        rgb["R"] = 0
    if rgb["G"] > 0:
        rgb["G"] = int(max_intensity * ((rgb["G"] * factor) ** gamma))
    else:
        rgb["G"] = 0
    if rgb["B"] > 0:
        rgb["B"] = int(max_intensity * ((rgb["B"] * factor) ** gamma))
    else:
        rgb["B"] = 0
    
    # Display no color as gray
    if (rgb["R"] + rgb["G"] + rgb["B"]) == 0:
        rgb["R"] = 155
        rgb["G"] = 155
        rgb["B"] = 155
    
    return (rgb["R"], rgb["G"], rgb["B"])


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


def readcal(width, cal_file='caldata.txt'):
    """
    Read calibration data and compute wavelength array.
    Returns: [wavelengthData, calmsg1, calmsg2, calmsg3]
    """
    errors = 0
    message = 0
    
    try:
        with open(cal_file, 'r') as file:
            lines = file.readlines()
            line0 = lines[0].strip()
            pixels = line0.split(',')
            pixels = [int(i) for i in pixels]
            line1 = lines[1].strip()
            wavelengths = line1.split(',')
            wavelengths = [float(i) for i in wavelengths]
    except:
        errors = 1

    if errors == 0:
        try:
            if len(pixels) != len(wavelengths):
                errors = 1
            if len(pixels) < 3:
                errors = 1
            if len(wavelengths) < 3:
                errors = 1
        except:
            errors = 1

    if errors == 1:
        # Default placeholder data
        pixels = [0, 400, 800]
        wavelengths = [380, 560, 750]

    wavelengthData = []

    if len(pixels) == 3:
        # Second order polynomial
        coefficients = np.poly1d(np.polyfit(pixels, wavelengths, 2))
        C1 = coefficients[2]
        C2 = coefficients[1]
        C3 = coefficients[0]
        
        for pixel in range(width):
            wavelength = ((C1 * pixel**2) + (C2 * pixel) + C3)
            wavelength = round(wavelength, 6)
            wavelengthData.append(wavelength)
        
        if errors == 1:
            message = 0  # Errors
        else:
            message = 1  # Only 3 wavelength cal (inaccurate)

    if len(pixels) > 3:
        # Third order polynomial
        coefficients = np.poly1d(np.polyfit(pixels, wavelengths, 3))
        C1 = coefficients[3]
        C2 = coefficients[2]
        C3 = coefficients[1]
        C4 = coefficients[0]
        
        for pixel in range(width):
            wavelength = ((C1 * pixel**3) + (C2 * pixel**2) + (C3 * pixel) + C4)
            wavelength = round(wavelength, 6)
            wavelengthData.append(wavelength)

        # Calculate R-squared
        predicted = []
        for i in pixels:
            px = i
            y = ((C1 * px**3) + (C2 * px**2) + (C3 * px) + C4)
            predicted.append(y)

        corr_matrix = np.corrcoef(wavelengths, predicted)
        corr = corr_matrix[0, 1]
        R_sq = corr**2
        
        print(f"R-Squared={R_sq}")
        message = 2  # Multi-wavelength cal, 3rd order poly

    if message == 0:
        calmsg1 = "UNCALIBRATED!"
        calmsg2 = "Defaults loaded"
        calmsg3 = "Perform Calibration!"
    elif message == 1:
        calmsg1 = "Calibrated!!"
        calmsg2 = "Using 3 cal points"
        calmsg3 = "2nd Order Polyfit"
    elif message == 2:
        calmsg1 = "Calibrated!!!"
        calmsg2 = "Using > 3 cal points"
        calmsg3 = "3rd Order Polyfit"

    return [wavelengthData, calmsg1, calmsg2, calmsg3]


def writecal(pixel_wavelength_pairs, cal_file='caldata.txt'):
    """
    Write calibration data to file.
    pixel_wavelength_pairs: list of tuples [(pixel, wavelength), ...]
    Returns: True if successful
    """
    if not pixel_wavelength_pairs or len(pixel_wavelength_pairs) < 3:
        print("Need at least 3 calibration points!")
        return False
    
    try:
        pxdata = [str(p[0]) for p in pixel_wavelength_pairs]
        wldata = [str(p[1]) for p in pixel_wavelength_pairs]
        
        # Validate wavelengths are numeric
        _ = [float(x) for x in wldata]
        
        pxdata_str = ','.join(pxdata)
        wldata_str = ','.join(wldata)
        
        with open(cal_file, 'w') as f:
            f.write(pxdata_str + '\r\n')
            f.write(wldata_str + '\r\n')
        
        print("Calibration Data Written!")
        return True
    except:
        print("Calibration write failed!")
        return False


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
