import neurokit2 as nk
from tqdm import tqdm
import os
import wfdb
import argparse
import numpy as np
import shutil
from collections import Counter

parser = argparse.ArgumentParser(description='Create dataset for MIT-BIH')
parser.add_argument('--data_folder', type=str, default='/media/Volume/data/MIT-BHI/data/', help='Path to raw data folder')
parser.add_argument('--hb_split_type', type=str, default='t_wave', help='How to split the heartbeats, either t_wave or static')
parser.add_argument('--name', type=str, default='t_wave_split', help='Name of the split')
parser.add_argument('--nk_clean', action='store_true', help='Use neurokit2 to clean the signal')
args = parser.parse_args()


train = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
test = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

def convert_label(symbol):
    if symbol in ['N', 'L', 'R', 'e', 'j']: 
        return 'N'
    elif symbol in ['A', 'a', 'J', 'S']:
        return 'S'  # Supraventricular ectopic
    elif symbol in ['V', 'E']: 
        return 'V'  # Ventricular ectopic
    elif symbol in ['F']: 
       return 'F'  # Fusion
    elif symbol in ['/', 'f', 'Q']:
        return 'Q'  # Unknown
    else:
        print(symbol)
        raise(f'Unknown symbol {symbol}') 
    

   
def crete_csv_mapping(patient_ids, data_folder, split='train', hb_split_type='t_wave', name='t_wave_split'):
    all_labels = []

    for patient in tqdm(patient_ids):
        signal, _ = wfdb.rdsamp(os.path.join(data_folder + 'raw', f'{patient}'))
        signal = signal[:, 0] # consider only one channel
        annotation = wfdb.rdann(os.path.join(data_folder + 'raw', f'{patient}'), 'atr')

        r_peaks = annotation.sample
        labels = annotation.symbol

        counter_labels = Counter(labels)
        print(f'Patient {patient} labels:', counter_labels)

        valid_annotations = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']

        # filter out the annotations that are not in the valid_annotations
        r_peaks = [r_peak for i, r_peak in enumerate(r_peaks) if labels[i] in valid_annotations]
        labels = [label for label in labels if label in valid_annotations]

        # find the T waves offsets
        if hb_split_type == 't_wave':
            cleaned_signal = nk.ecg_clean(signal, sampling_rate=360)
            _, waves_peak = nk.ecg_delineate(
                cleaned_signal, 
                rpeaks=r_peaks, 
                method='dwt',
                sampling_rate=360,
                show=False,
            )
            
            t_offsets = waves_peak['ECG_T_Offsets']
            # if any of this has a nan value print
            print('Count of NaN elements in t_offsets:', np.isnan(t_offsets).sum())

        file_path = os.path.join(data_folder + name, f'labels_{split}.csv')
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write('patient,sample_id,orig_label,label,hb_start,hb_end,r_peak,is_oversampled,win_start,win_end\n')

        with open(file_path, 'a') as f:
            start = 0
            end = 0
            for i, (r_p, t_o) in enumerate(zip(r_peaks, t_offsets)):
                prev_end = end
                # if the last element, consider the end of the signal
                if i == len(r_peaks) - 1:
                    end = len(signal)
                elif np.isnan(t_o):
                    # consider the half of the distance between the current r_p and the next r_p
                    end = (r_p + r_peaks[i + 1]) // 2
                else:
                    # consider the T wave offset
                    end = t_o

                start = prev_end

                # ensure that the window is maximum 600 samples on both sides
                if r_p - start > 400:
                    start = r_p - 400
                    print(f'found an heartbeat with more than 400 samples before the r_p, cutting it, patient {patient}, sample {i}')
                if end - r_p > 400:
                    end = r_p + 400
                    print(f'found an heartbeat with more than 400 samples after the r_p, cutting it, patient {patient}, sample {i}')

                class_label = convert_label(labels[i])

                # All labels: Counter({'N': 45866, 'V': 3788, 'S': 944, 'F': 415, 'Q': 8})
                # We want to oversample the minority classes
                # V will have 2 different windows
                # S will have 4 differet windows
                # F will have 8 different windows
                # Q will have 10 different windows

                def get_centered_window(signal, r_p, window_size, start, end):
                    # start at zero or at the minimum value between the start and the r_p - window_size // 2
                    start_window = max(0, min(start, r_p - window_size // 2))
                    end_window = min(len(signal), max(end, r_p + window_size // 2))
                    return start_window, end_window

                def get_other_window(signal, r_p, window_size, num_windows):
                    windows = []
                    shift_size = window_size // (num_windows + 1)
                    # add before
                    for i in range(num_windows):
                        left_side = shift_size * (i + 1)
                        start_window = max(0, min(start, r_p - left_side))
                        end_window = min(len(signal), max(end, r_p + window_size - left_side))
                        windows.append((start_window, end_window))
                    return windows
                        

                # consider a 10s window
                window_size = 10 * 360
                start_window, end_window = get_centered_window(signal, r_p, window_size, start, end)
                f.write(f'{patient},{i},{labels[i]},{class_label},{start},{end},{r_p},False,{start_window},{end_window}\n')

                all_labels.append(class_label)
                if split == 'test':
                    continue

                window_to_add = 0
                if class_label == 'V':
                    window_to_add = 2
                elif class_label == 'S':
                    window_to_add = 6
                elif class_label == 'F':
                    window_to_add = 8
                elif class_label == 'Q':
                    window_to_add = 12

                for start_window, end_window in get_other_window(signal, r_p, window_size, window_to_add):
                    all_labels.append(class_label)
                    f.write(f'{patient},{i},{labels[i]},{class_label},{start},{end},{r_p},True,{start_window},{end_window}\n')



    count_all_labels = Counter(all_labels)
    print(f'All labels:', count_all_labels)


if __name__ == '__main__':
        # remove the folder if it exists and create a new one
    if os.path.exists(args.data_folder + args.name):
        shutil.rmtree(args.data_folder + args.name)
        
    os.makedirs(args.data_folder + args.name, exist_ok=True)
    crete_csv_mapping(train, args.data_folder, 'train', name=args.name)
    crete_csv_mapping(test, args.data_folder, 'test', name=args.name)



