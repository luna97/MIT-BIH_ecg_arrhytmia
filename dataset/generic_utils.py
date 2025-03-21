import numpy as np
import os 
from tqdm import tqdm
import torch
from joblib import Parallel, delayed

def get_max_n_jobs(default=-1):
    n_jobs = int(os.getenv("SLURM_CPUS_PER_TASK", default))
    return n_jobs
    

def random_shift(signal, patch_size):
    # remove a random number of datapoints from the signal from 0 to patch size 
    shift = np.random.randint(0, patch_size // 2)
    if shift > 0 and len(signal) > shift:
        signal = signal[shift:]
    return signal

    
def find_records(folder, header_extension='.dat'):
    def process_file(root, file):
        extension = os.path.splitext(file)[1]
        if extension == header_extension:
            record = os.path.relpath(os.path.join(root, file), folder)[:-len(header_extension)]
            return record
        return None

    records = set()

    print(f'Finding records in {folder}...')
    results = Parallel(n_jobs=get_max_n_jobs())(delayed(process_file)(root, file) for root, _, files in os.walk(folder) for file in files)
    records.update(filter(None, results))
    records = sorted(records)
    return records

def check_mean_var_r_peaks(sample):
    if torch.isnan(sample['r_peak_interval_mean']):
        sample['r_peak_interval_mean'] = torch.tensor(0)
    if torch.isnan(sample['r_peak_variance']):
        sample['r_peak_variance'] = torch.tensor(0)  
    return sample