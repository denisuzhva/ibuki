import matplotlib.pyplot as plt



def plot_waveforms(waveform_list):
    n_cols = len(waveform_list)
    fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(15, 2))
    for axis, waveform in zip(axes, waveform_list):
        axis.plot(waveform)
    plt.show()