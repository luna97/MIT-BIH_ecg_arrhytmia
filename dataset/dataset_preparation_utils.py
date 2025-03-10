import wfdb
import neurokit2 as nk
import numpy as np
import os
import shutil
import h5py

lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def unpad_signal(signal):
    start_unpad_idx = 0
    while start_unpad_idx < signal.shape[0] and np.all(signal[start_unpad_idx, :] == 0):
        start_unpad_idx += 1

    end_unpad_idx = signal.shape[0]
    while end_unpad_idx > start_unpad_idx and np.all(signal[end_unpad_idx-1, :] == 0):
        end_unpad_idx -= 1

    if start_unpad_idx >= end_unpad_idx:
        return None
    else:
        return signal[start_unpad_idx:end_unpad_idx, :]
    

def process_sample(signal, output_file_path, record_name, fs, desired_fs, nk_clean=False):
    # create out directory if not exists
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    signal = unpad_signal(signal)
    if signal is None:
        print(f"Record {record_name} has only zeros - skipping")
        return None

    for i in range(signal.shape[1]):
        signal[:, i] = nk.signal_fillmissing(signal[:, i], method='both')
        if nk_clean:
            signal[:, i] = nk.ecg_clean(signal[:, i], sampling_rate=desired_fs)

    # resample the signal to 360 hz
    signal = nk.signal_resample(signal, sampling_rate=fs, desired_sampling_rate=desired_fs, method='FFT')

    if len(signal) < 360:
        print(f"Record {record_name} has less than 360 samples - skipping")
        return None

    # skip record with too big variance and high values
    if np.var(signal) > 10 and (np.max(signal[:, i]) >= 15 or np.min(signal[:, i]) < -15):
        print(f"Record {record_name} has too high variance - skipping")
        return None
    if np.var(signal) < 0.0001:
        print(f"Record {record_name} has too low variance - skipping")
        return None
    
    return signal

def resample_and_save_record_wfdb(record_path, desired_fs, output_file_path, nk_clean=False):
    record = wfdb.rdrecord(record_path)

    signal = process_sample(record.p_signal, output_file_path, record.record_name, record.fs, desired_fs, nk_clean)
        
    if signal is None:
        return

    wfdb.wrsamp(
        record.record_name,
        fs=desired_fs,
        units=record.units, 
        sig_name=record.sig_name, 
        samps_per_frame=record.samps_per_frame,
        p_signal=signal, 
        fmt=record.fmt, 
        adc_gain=record.adc_gain, 
        baseline=record.baseline, 
        comments=record.comments,
        base_date=record.base_date,
        base_time=record.base_time,
        write_dir=output_file_path, # output here
    )

def resample_and_save_record_hdf5(record_path, exam_id, desired_fs, output_file_path, nk_clean=False):
    with h5py.File(record_path, 'r') as f:
        # idx of the exam_id
        signal_idx = np.where(f['exam_id'][:] == exam_id)[0][0]
        signal = np.array(f['tracings'][signal_idx], dtype=np.float32)

        signal = process_sample(signal, output_file_path, exam_id, 500, desired_fs, nk_clean)

        if signal is None:
            return
        
        try:
            wfdb.wrsamp(
                str(exam_id),
                fs=desired_fs,
                units=['mV']*12, 
                sig_name=lead_names, 
                p_signal=signal, 
                fmt=['16']*len(lead_names), 
                adc_gain=[1000]*12, 
                baseline=[0]*12,
                write_dir=output_file_path, # output here
            )
        except Exception as e:
            print(f"Error in record {exam_id}: {e}")

def clean_and_create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        shutil.rmtree(directory)
        os.makedirs(directory)


# Fix the checksums from the Python WFDB library.
def fix_checksums(record, checksums=None):
    if checksums is None:
        x = wfdb.rdrecord(record, physical=False)   
        signals = np.asarray(x.d_signal)
        checksums = np.sum(signals, axis=0, dtype=np.int16)

    header_filename = os.path.join(record + '.hea')
    string = ''
    with open(header_filename, 'r') as f:
        for i, l in enumerate(f):
            if i == 0:
                arrs = l.split(' ')
                num_leads = int(arrs[1])
            if 0 < i <= num_leads and not l.startswith('#'):
                arrs = l.split(' ')
                arrs[6] = str(checksums[i-1])
                l = ' '.join(arrs)
            string += l

    with open(header_filename, 'w') as f:
        f.write(string)