# fbs.py - FBS customization
# RMM, 8 Oct 2021

import matplotlib.pyplot as plt

# Set the fonts to match the main text
plt.rc('font', family='Times New Roman', weight='normal', size=14)
plt.rcParams['mathtext.fontset'] = 'cm'


# Create a new figure of a specified size
def figure(size='mlh'):
    if size == 'mlh' or size == '221':
        plt.figure(figsize=[3.4, 2.55])
    else:
        raise ValueError("unknown figure size")
    

# Print a figure
def savefig(name, pad=0.1, **kwargs):
    plt.tight_layout(pad=pad)   # clean up plots
    plt.savefig(name)           # save to file
