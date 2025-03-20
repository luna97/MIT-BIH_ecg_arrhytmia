import wfdb
import neurokit2 as nk
import numpy as np
import os
import shutil
import h5py
import json
import simple_icd_10

lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# icd10 codes and their classes
rheumatic = ['I00', 'I01', 'I05', 'I06', 'I07', 'I08', 'I09']
hypertensive_codes = ['I10', 'I11', 'I12', 'I13', 'I15']
ischaemic_codes = ['I20', 'I21', 'I22', 'I23', 'I24', 'I25']
pulmonary_heart = ['I26', 'I27', 'I28']
cerebro_vascular = ['I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69']
arteries_diseases = ['I70', 'I71', 'I72', 'I73', 'I74', 'I77', 'I78','I79']
veins_diseases = ['I80', 'I81', 'I82', 'I83', 'I85', 'I86', 'I87', 'I88', 'I89']
hypotension = ['I95']
heart_failure = ['I50']
cardiomiopathy = ['I42', 'I43']

# common classes
first_degree_avb = ['I440']
left_bundle_bb = ['I447']
right_bundle_bb = ['I451']
bradycardia = ['R001']
tachicardia = ['I47', 'R000']
atrial_fibrillation = ['I48']

def parse_diagnosis(row):
    """
    Parse the diagnosis from the row

    :param row: the row of the dataframe
    :return: the unique list of diagnosis
    """
    lists = [json.loads(l.replace("'", '"')) for l in row[['ed_diag_ed', 'ed_diag_hosp', 'hosp_diag_hosp', 'all_diag_hosp', 'all_diag_all']].values if l != '[]' and l != None]
    # put lists in a single list
    tortn = [item[:4] for sublist in lists for item in sublist if item.startswith('I') or item.startswith('R00')]
    # return unique values
    return json.dumps(list(set(tortn)))

def startswith_any(codes, startswith):
    """
    Check if the code starts with any of the strings in the list
    """
    for s in startswith:
        if codes.startswith(s):
            return True
    return False

def get_group(row):
    """
    Get the group of the diagnosis, aligned with the most common classes
    
    :param row: the row of the dataframe
    :return: the group of the diagnosis
    """
    code = row['icd10_codes_set']
    codes = json.loads(code)
    tortn = []

    for c in codes:
        if startswith_any(c, rheumatic):
            tortn.append('Rheumatic disease')
        elif startswith_any(c, hypertensive_codes):
            tortn.append('Hypertension')
        elif startswith_any(c, ischaemic_codes):
            tortn.append('Ischaemic disease')
        elif startswith_any(c, pulmonary_heart):
            tortn.append('Pulmonary Heart')
        elif startswith_any(c, cerebro_vascular):
            tortn.append('Cerebrovascular diseases')
        elif startswith_any(c, arteries_diseases):
            tortn.append('Arteries diseases')
        elif startswith_any(c, veins_diseases):
            tortn.append('Veins diseases')
        elif startswith_any(c, hypotension):
            tortn.append('Hypotension')
        elif startswith_any(c, heart_failure):
            tortn.append('Heart Failure')
        elif startswith_any(c, cardiomiopathy):
            tortn.append('Cardiomiopathy')
        # needed for labels
        elif startswith_any(c, first_degree_avb):
            tortn.append('1dAVB')
        elif startswith_any(c, atrial_fibrillation):
            tortn.append('AF')
        elif startswith_any(c, tachicardia):
            tortn.append('ST')
        elif startswith_any(c, bradycardia):
            tortn.append('SB')
        elif startswith_any(c, left_bundle_bb):
            tortn.append('LBBB')
        elif startswith_any(c, right_bundle_bb):
            tortn.append('RBBB')
        else:
            tortn.append('Other')
        return json.dumps(list(set(tortn)))
    
def get_desc(row):
    """
    Get the description of the diagnosis
    
    :param row: the row of the dataframe
    :return: the description of the diagnosis
    """
    code = row['icd10_codes_set']
    codes = json.loads(code)
    tortn = []
    for c in codes:
        try:
            tortn.append(simple_icd_10.get_description(c[:3]) + ' (' + c[:3] + ')')
        except:
            tortn.append(c)
    return json.dumps(tortn)

def get_array(row):
    """
    Get the array of the diagnosis
    """
    if row['icd10_codes_groups'] is None:
        return []
    array = json.loads(row['icd10_codes_groups'])
    if array is None:
        return []
    return array


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
    if fs != desired_fs:
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