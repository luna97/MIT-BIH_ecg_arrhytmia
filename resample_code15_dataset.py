import argparse
import os
from dataset.helper_code_code15 import find_records, load_label, load_signals
from neurokit2 import signal_resample
from tqdm import tqdm
import numpy as np
import wfdb
from joblib import Parallel, delayed
from neurokit2 import ecg_clean  

parser = argparse.ArgumentParser(description='Prepare the CODE-15 database.')
parser.add_argument('-df','--data_folder', type=str, required=False, default='/media/Volume/data/CODE15/')
parser.add_argument('-o', '--output_path', type=str, default='/media/Volume/data/CODE15/unlabeled_records_360/')
parser.add_argument('-l', '--leads', type=str, required=True, nargs='*')
parser.add_argument('-n', '--nk_clean', action='store_true')
args = parser.parse_args()

lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
sampling_frequency = 400
units = 'mV'
# Define the paramters for the WFDB files.
gain = 1000
baseline = 0
num_bits = 16
fmt = str(num_bits)
num_leads = 12


def resample_signal(record, folder, output_path, leads, nk_clean):
    signal, _ = load_signals(folder + 'unlabeled_records/' + record)
    # print("Original shape:", signal.shape)

    # if the first lead is zero, skip
    resampled = []
    for l in leads:  # Use actual number of leads
        i = lead_names.index(l)
        if np.all(signal[:, i] == 0):
            print(f"Record {record} has zeros only for lead {l} and length {signal.shape[0]} - skipping")
            return
        if len(signal[:, i]) < 128:
            print(f"Record {record} has less than 128 samples for lead {l} - skipping")
            return
        # if a sample has nan, skip
        if np.isnan(signal[:,i]).any():
            print(f"Record {record} has nan for lead {l} - skipping")
            return

    # now i can resample if is everything is okay
    for l in leads:  # Use actual number of leads
        i = lead_names.index(l)
        sample = signal_resample(signal[:,i], sampling_rate=400, desired_sampling_rate=360, method='FFT')
        if nk_clean:
            sample = ecg_clean(sample, sampling_rate=360)
        resampled.append(sample)
        if np.all(sample == 0):
            print(f"Resampled record {record} has resampled signal of length {len(resampled[-1])} for lead {l} - skipping")
            return
        if len(sample) < 128:
            print(f"Resampled record {record} has less than 128 samples for lead {l} - skipping")
            return

    # Stack the resampled signals correctly
    signal = np.stack(resampled, axis=1)  # This gives (time_points, leads)
    # print("Resampled shape:", signal.shape)

    # if signal has nan return
    if np.isnan(signal).any():
        print(f"Record {record} has nan after resampling - skipping")
        return

    # set nan to 0
    signal = np.nan_to_num(signal)

    # Convert to int32 if needed
    signal = (signal * gain).astype(np.int16)    
    wfdb.wrsamp(
        record_name=record, 
        fs=360, 
        units=[units]*signal.shape[1],  # Use actual number of leads
        sig_name=lead_names[:signal.shape[1]],  # Use actual number of leads
        d_signal=signal, 
        fmt=[fmt]*signal.shape[1],  # Use actual number of leads
        adc_gain=[gain]*signal.shape[1],  # Use actual number of leads
        baseline=[baseline]*signal.shape[1],  # Use actual number of leads
        write_dir=output_path,
        comments=[]
    )

def normalize_code15_dataset(folder, output_path, leads, nk_clean):
    # delete everything in the out dir
    os.system(f'rm -rf {output_path}/*')
    # mkdir out folder
    os.makedirs(output_path, exist_ok=True)
    records = find_records(folder + 'unlabeled_records')
    Parallel(n_jobs=-1)(delayed(resample_signal)(record, folder, output_path, leads, nk_clean) for record in records)


normalize_code15_dataset(args.data_folder, args.output_path, args.leads, args.nk_clean)

