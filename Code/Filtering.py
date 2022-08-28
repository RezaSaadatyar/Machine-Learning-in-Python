from scipy import signal


def filtering(Data, F_low, F_high, Order, Fs, btype):
    if Data.shape[0] > Data.shape[1]:
        data = Data.T
    f_low = F_low / (Fs / 2)
    f_high = F_high / (Fs / 2)
    if btype == "low":
        b, a = signal.butter(Order, f_low, btype='low')
    elif btype == "high":
        b, a = signal.butter(Order, f_high, btype='high')
    elif btype == "bandpass":
        b, a = signal.butter(Order, [f_low, f_high], btype='bandpass')
    elif btype == "bandstop":
        b, a = signal.butter(Order, [f_low, f_high], btype='bandstop')
    data_filter = signal.filtfilt(b, a, data)
    return data_filter.T
