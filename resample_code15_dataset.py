import argparse
import os
from dataset.helper_code_code15 import find_records, load_label, load_signals
from neurokit2 import signal_resample
from tqdm import tqdm
import numpy as np
import wfdb
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='Prepare the CODE-15 database.')
parser.add_argument('-df','--data_folder', type=str, required=False, default='/media/Volume/data/CODE15/')
parser.add_argument('-o', '--output_path', type=str, default='/media/Volume/data/CODE15/records/')
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


def resample_signal(record, folder, output_path):
    signal, _ = load_signals(folder + 'unlabeled_records/' + record)
    # print("Original shape:", signal.shape)
    
    resampled = []
    for i in range(signal.shape[1]):  # Use actual number of leads
        resampled.append(signal_resample(signal[:,i], sampling_rate=400, desired_sampling_rate=360))

    # Stack the resampled signals correctly
    signal = np.stack(resampled, axis=1)  # This gives (time_points, leads)
    # print("Resampled shape:", signal.shape)
    
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

def normalize_code15_dataset(folder, output_path):
    # delete everything in the out dir
    os.system(f'rm -rf {output_path}/*')
    # mkdir out folder
    os.makedirs(output_path, exist_ok=True)
    records = find_records(folder + 'unlabeled_records')
    Parallel(n_jobs=-1)(delayed(resample_signal)(record, folder, output_path) for record in records)


normalize_code15_dataset(args.data_folder, args.output_path)
