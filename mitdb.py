import os
import h5py
import wfdb as wf
import numpy as np
import pandas as pd
from glob import glob
from scipy import signal as ss
from matplotlib import pyplot as plt

def get_records():
    """ Get paths for data in data/mit/ directory """

    # There are 3 files for each record
    # *.atr is one of them
    paths = glob('data/*.atr')

    # Get rid of the extension
    paths = [path[:-4] for path in paths]
    paths.sort()

    return paths

def good_types():
    """ Of annotations """
    # www.physionet.org/physiobank/annotations.shtml
    good = ['N', 'L', 'R', 'B', 'A',
            'a', 'J', 'S', 'V', 'r',
            'F', 'e', 'j', 'n', 'E',
            '/', 'f', 'Q', '?']

    return good

def beat_annotations(annotation):
    """ Get rid of non-beat markers """
    # Declare beat types
    good = good_types()
    ids = np.isin(annotation.anntype, good)

    # We want to know only the positions
    beats = annotation.annsamp[ids]

    return beats

def convert_input(channel, annotation):
    """ Into output """
    # Remove non-beat annotations
    beats = beat_annotations(annotation)

    # Create dirac-comb signal
    dirac = np.zeros_like(channel)
    dirac[beats] = 1.0

    # Use hamming window as a bell-curve filter
    width = 36
    filter = ss.hamming(width)
    gauss = np.convolve(filter, dirac, mode = 'same')

    return dirac, gauss

def good_annotations():
    """ Const function with good annotations """
    # For now it seems those are most popular
    good_annotations = [1, 2, 3, 4,
                        5, 6, 7, 8,
                        9, 10, 11, 12,
                        13, 16, 31, 38]

    return good_annotations

def make_dataset(records, width, savepath):
    """ Inside an array """
    # Prepare containers
    signals, labels = [], []

    # Iterate files
    for path in records:
        print ('Processing file:', path)
        record = wf.rdsamp(path)
        annotations = wf.rdann(path, 'atr')

        # Extract pure signals
        data = record.p_signals

        # Convert each channel into labeled fragments
        signal, label = convert_data(data, annotations, width)

        # Cumulate
        signals.append(signal)
        labels.append(label)

    # Convert to one huge numpy.array
    signals = np.vstack(signals)
    labels = np.vstack(labels)

    # Write to disk
    np.save(savepath, {'signals' : signals,
                       'labels'  : labels })

def convert_data(data, annotations, width):
    """ Into a batch """
    # Prepare containers
    signals, labels = [], []

    # Convert both channels
    for it in range(2):
        channel = data[:, it]
        dirac, gauss = convert_input(channel, annotations)
        # Merge labels
        label = np.vstack([dirac, gauss])

        # Prepare the moving window
        sta = 0
        end = width
        stride = width
        while end <= len(channel):
            # Chop out the fragments
            s_frag = channel[sta : end]
            l_frag = label[:, sta : end]

            # Cumulate
            signals.append(s_frag)
            labels.append(l_frag)

            # Go forth
            sta += stride
            end += stride

    # Turn into arrays
    signals = np.array(signals)
    labels = np.array(labels)

    return signals, labels

def create_datasets():
    """ Training, validation, test """

    # Define the data
    width = 200

    train = [101, 106, 108, 109, 114, 115, 116, 119, 122, 124, 201, 203, 208, 209, 215, 223, 230]
    validation = [112, 118, 205, 207, 220]
    # excluded = [102, 104, 107, 217]
    test = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


    # Make training
    make_dataset(train, width, 'data/training')

    # ... validation ...
    make_dataset(validation, width, 'data/validation')

    # ... and test
    make_dataset(validation, width, 'data/test')

