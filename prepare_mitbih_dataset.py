import neurokit2 as nk
from tqdm import tqdm
import os
import wfdb
import argparse
import numpy as np
import shutil
from collections import Counter
import pandas as pd
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='Create dataset for MIT-BIH')
parser.add_argument('--data_folder', type=str, default='/media/Volume/data/MIT-BHI/data/', help='Path to raw data folder')
parser.add_argument('--hb_split_type', type=str, default='t_wave', help='How to split the heartbeats, either t_wave or static')
parser.add_argument('--name', type=str, default='t_wave_split', help='Name of the split')
parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs. -1 means using all processors.')
args = parser.parse_args()

train = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
test = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


def convert_label(symbol):
    """
    Convert the symbols to the main class label
    """
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
        raise (f'Unknown symbol {symbol}')


def get_centered_window(signal, r_p, window_size, start, end):
    """
    Get a window of size window_size centered around the r_peak
    """
    start_window = max(0, min(start, r_p - window_size // 2))
    end_window = min(len(signal), max(end, r_p + window_size // 2))
    return start_window, end_window


def get_other_window(signal, r_p, window_size, num_windows, start, end):
    """
    Get num_windows windows around the r_peak
    """
    windows = []
    shift_size = window_size // (num_windows + 1)
    for i in range(num_windows):
        left_side = shift_size * (i + 1)
        start_window = max(0, min(start, r_p - left_side))
        end_window = min(len(signal), max(end, r_p + window_size - left_side))
        windows.append((start_window, end_window))
    return windows


def process_patient(patient, data_folder, split, hb_split_type, name):
    """
    Process a single patient's data.  This function is designed for parallel execution.
    """
    all_data = []
    valid_annotations = set(
        ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q'])

    signal, _ = wfdb.rdsamp(os.path.join(data_folder + 'raw', f'{patient}'))
    signal = signal[:, 0]
    annotation = wfdb.rdann(
        os.path.join(data_folder + 'raw', f'{patient}'), 'atr')
    header = wfdb.rdheader(os.path.join(data_folder + 'raw', f'{patient}'))
    comment = header.comments[0]
    try:
        age = int(comment.split(' ')[0])
    except:
        age = 0

    is_male = comment.split(' ')[1] == 'M'

    r_peaks = annotation.sample
    labels_orig = annotation.symbol

    # Filter annotations
    r_peaks = [r_peak for i, r_peak in enumerate(
        r_peaks) if labels_orig[i] in valid_annotations]
    labels = [label for label in labels_orig if label in valid_annotations]

    annotation_positions = [i for i, label in enumerate(
        labels_orig) if label == '+']
    extra_labels = [label.removesuffix('\x00').removeprefix('(')
                    for label in annotation.aux_note if label.startswith('(')]

    # T-wave delineation (only if needed)
    t_offsets = [np.nan] * len(r_peaks)  # Initialize with NaNs
    if hb_split_type == 't_wave':
        cleaned_signal = nk.ecg_clean(signal, sampling_rate=360)
        # Handle potential errors during delineation
        try:
            _, waves_peak = nk.ecg_delineate(
                cleaned_signal,
                rpeaks=r_peaks,  # Use filtered r_peaks
                method='dwt',
                sampling_rate=360,
                show=False,
            )
            t_offsets = waves_peak['ECG_T_Offsets']
             # Ensure t_offsets has the same length as filtered r_peaks
            if len(t_offsets) < len(r_peaks):
                t_offsets = np.concatenate([t_offsets, np.full(len(r_peaks)-len(t_offsets), np.nan)]) # Pad with NaNs
        except Exception as e:
            print(f"Error delineating patient {patient}: {e}")
            # Keep t_offsets as all NaNs, processing will continue with median strategy

    start = 0
    end = 0
    for i, (r_p, t_o) in enumerate(zip(r_peaks, t_offsets)):
        prev_end = end
        if i == len(r_peaks) - 1:
            end = len(signal)
        elif np.isnan(t_o):
            end = (r_p + r_peaks[i + 1]) // 2 if i + \
                1 < len(r_peaks) else len(signal)  # Handle last peak
        else:
            end = int(t_o)  # Ensure integer index

        start = prev_end

        start = max(0, int(start))  # Ensure valid indices
        end = max(0, int(end))

        if r_p - start > 400:
            start = r_p - 400
        if end - r_p > 400:
            end = r_p + 400
        
        start = max(0, int(start))  # Ensure valid indices
        end = max(0, int(end))
        class_label = convert_label(labels[i])

        def get_extra_annotation(start, end, annotation_positions, extra_labels):
            anns = []
            for i, pos in enumerate(annotation_positions):
                if start < pos <= end:
                    anns.append(extra_labels[i].strip())
            return '_'.join(anns) if anns else ''

        window_size = 10 * 360
        start_window, end_window = get_centered_window(
            signal, r_p, window_size, start, end)
        extra_annotations = get_extra_annotation(
            start_window, end_window, annotation_positions, extra_labels)
        r_peaks_in_window = np.array([ r for r in r_peaks if start_window <= r <= end_window])

        row_data = {
            'patient': patient,
            'sample_id': i,
            'orig_label': labels[i],
            'label': class_label,
            'hb_start': start,
            'hb_end': end,
            'r_peak': r_p,
            'is_oversampled': False,
            'win_start': start_window,
            'win_end': end_window,
            'age': age,
            'is_male': is_male,
            'extra_annotations': extra_annotations,
            'r_peaks_interval_mean': np.mean((r_peaks_in_window[1:] - r_peaks_in_window[:-1]) / 360),
            'r_peak_variance': np.std((r_peaks_in_window[1:] - r_peaks_in_window[:-1]) / 360),
        }
        all_data.append(row_data)


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

        for start_window, end_window in get_other_window(signal, r_p, window_size, window_to_add, start, end):
            extra_annotations = get_extra_annotation(
                start_window, end_window, annotation_positions, extra_labels)
            r_peaks_in_window = np.array([ r for r in r_peaks if start_window <= r <= end_window])

            row_data = {
                'patient': patient,
                'sample_id': i,  # Use original sample_id
                'orig_label': labels[i],
                'label': class_label,
                'hb_start': start,
                'hb_end': end,
                'r_peak': r_p,
                'is_oversampled': True,
                'win_start': start_window,
                'win_end': end_window,
                'age': age,
                'is_male': is_male,
                'extra_annotations': extra_annotations,
                'r_peaks_interval_mean': np.mean((r_peaks_in_window[1:] - r_peaks_in_window[:-1]) / 360),
                'r_peak_variance': np.std((r_peaks_in_window[1:] - r_peaks_in_window[:-1]) / 360),
            }
            all_data.append(row_data)

    return all_data


def create_csv_mapping(patient_ids, data_folder, split='train', hb_split_type='t_wave', name='t_wave_split', n_jobs=-1):
    """
    Create CSV mapping with parallel processing.
    """

    # Use joblib.Parallel to process patients in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_patient)(patient, data_folder, split, hb_split_type, name)
        for patient in tqdm(patient_ids, desc=f"Processing {split} data")
    )

    # Flatten the list of lists (results from each patient)
    all_data = [item for sublist in results for item in sublist]

    df = pd.DataFrame(all_data)
    file_path = os.path.join(data_folder + name, f'labels_{split}.csv')
    df.to_csv(file_path, index=False)

    all_labels = df['label'].tolist()  # Efficiently get all labels from DataFrame
    count_all_labels = Counter(all_labels)
    print(f'All {split} labels:', count_all_labels)



if __name__ == '__main__':
    if os.path.exists(args.data_folder + args.name):
        shutil.rmtree(args.data_folder + args.name)

    os.makedirs(args.data_folder + args.name, exist_ok=True)
    create_csv_mapping(train, args.data_folder, 'train',
                       name=args.name, n_jobs=args.n_jobs)
    create_csv_mapping(test, args.data_folder, 'test',
                       name=args.name, n_jobs=args.n_jobs)