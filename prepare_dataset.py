import neurokit2 as nk
from tqdm import tqdm
import os
import wfdb
import argparse
import numpy as np
import shutil
import pandas as pd
from collections import Counter
from dataset.dataset_preparation_utils import *
from joblib import Parallel, delayed
import json
import wfdb.processing as wp
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)


parser = argparse.ArgumentParser(description='Create dataset for MIT-BIH')
parser.add_argument('--data_folder', type=str, default='/media/Volume/data/MIMIC_IV/', help='Path to raw data folder')
parser.add_argument('--label_file', type=str, default='/media/Volume/data/MIMIC_IV/records_w_diag_icd10.csv', help='Path to the label file')
parser.add_argument('--nk_clean', action='store_true', help='Use NeuroKit2 to clean the data')
parser.add_argument('--output_folder', type=str, required=True, help='the name of the output directory')
parser.add_argument('--dataset', type=str, required=True, help='the name of the dataset: mimic or code 15')
args = parser.parse_args()

def process_sample_mimic(sample):
    record_path = os.path.join(args.data_folder, 'files', 'p' + str(sample['subject_id'])[:4], 'p' + str(sample['subject_id']), 's' + str(sample['study_id']), str(sample['study_id']))
    out_path = os.path.join(args.output_folder, str(sample['study_id']))
    resample_and_save_record_wfdb(record_path, 360, out_path, nk_clean=args.nk_clean)

def add_labels_mimic(exams):
    exams['icd10_codes_set'] = exams.parallel_apply(lambda row: parse_diagnosis(row), axis=1)
    exams['icd10_codes_descs'] = exams.parallel_apply(lambda row: get_desc(row), axis=1)
    exams['icd10_codes_groups'] = exams.parallel_apply(lambda row: get_group(row), axis=1)

    # set the same categories as in the code 15 dataset
    exams['1dAVB'] = exams.parallel_apply(lambda row: '1dAVB' in get_array(row), axis=1)
    exams['RBBB'] = exams.parallel_apply(lambda row: 'RBBB' in get_array(row), axis=1)
    exams['LBBB'] = exams.parallel_apply(lambda row: 'LBBB' in get_array(row), axis=1)
    exams['SB'] = exams.parallel_apply(lambda row: 'SB' in get_array(row), axis=1)
    exams['ST'] = exams.parallel_apply(lambda row: 'ST' in get_array(row), axis=1)
    exams['AF'] = exams.parallel_apply(lambda row: 'AF' in get_array(row), axis=1)

    # add all the other labels
    exams['Other'] = exams.parallel_apply(lambda row: 'Other' in get_array(row), axis=1)
    exams['Hypertension'] = exams.parallel_apply(lambda row: 'Hypertension' in get_array(row), axis=1)
    exams['Ischaemic disease'] = exams.parallel_apply(lambda row: 'Ischaemic disease' in get_array(row), axis=1)
    exams['Pulmonary Heart'] = exams.parallel_apply(lambda row: 'Pulmonary Heart' in get_array(row), axis=1)
    exams['Cerebrovascular diseases'] = exams.parallel_apply(lambda row: 'Cerebrovascular diseases' in get_array(row), axis=1)
    exams['Arteries diseases'] = exams.parallel_apply(lambda row: 'Arteries diseases' in get_array(row), axis=1)
    exams['Veins diseases'] = exams.parallel_apply(lambda row: 'Veins diseases' in get_array(row), axis=1)
    exams['Hypotension'] = exams.parallel_apply(lambda row: 'Hypotension' in get_array(row), axis=1)
    exams['Heart Failure'] = exams.parallel_apply(lambda row: 'Heart Failure' in get_array(row), axis=1)
    exams['Cardiomiopathy'] = exams.parallel_apply(lambda row: 'Cardiomiopathy' in get_array(row), axis=1)
    exams['Rheumatic disease'] = exams.parallel_apply(lambda row: 'Rheumatic disease' in get_array(row), axis=1)
    return exams

def analyze_ecg(path):
    """
    Analyze the ecg signal

    :param row: the row of the dataframe
    :return: the peaks
    """
    # check path
    if not os.path.exists(path + '.dat'):
        print(f'Path {path} does not exist')
        return None
    
    record = wfdb.rdrecord(path)
    peaks = wp.xqrs_detect(record.p_signal[:, 1], record.fs, verbose=False)
    return peaks

def add_signal_informations(exams, identifier_key='study_id'):
    """
    Add some signal informations to the exams dataframe like r peak frequency and variance

    :param exams: the exams dataframe
    :return: the exams dataframe with the peaks informations
    """
    exams['r_peaks'] = exams.parallel_apply(lambda row: analyze_ecg(os.path.join(args.output_folder, str(row[identifier_key]))), axis=1)
    exams['r_peak_interval_mean'] = exams.parallel_apply(lambda row: np.mean((row['r_peaks'][1:] - row['r_peaks'][:-1]) / 360) if row['r_peaks'] is not None else None, axis=1)
    exams['r_peak_variance'] = exams.parallel_apply(lambda row: np.std((row['r_peaks'][1:] - row['r_peaks'][:-1]) / 360) if row['r_peaks'] is not None else None, axis=1)
    return exams


def process_csv_file_mimic(csv_file, out_csv_file):
    """
    Transform the icd10 codes into a ready to use format aligned with common classes
    Add r peak informations to the csv file

    :param csv_file: the path to the csv file
    :param out_csv_file: the path to the output csv file
    :return:
    """
    exams = pd.read_csv(csv_file)
    # drop useless columns
    exams.drop(columns=['ecg_no_within_stay', 'ecg_no_within_stay', 'ecg_taken_in_hosp', 'ecg_taken_in_ed_or_hosp', 'anchor_year', 'anchor_age'], inplace=True)
    
    exams = add_labels_mimic(exams)
    exams = add_signal_informations(exams, identifier_key='study_id')

    exams.to_csv(out_csv_file)

def process_csv_file_code15(csv_file, out_csv_file):
    """
    Add r peaks informations to the csv file

    :param csv_file: the path to the csv file
    :param out_csv_file: the path to the output csv file
    :return:
    """
    exams = pd.read_csv(csv_file)

    exams = add_signal_informations(exams, identifier_key='exam_id')
    exams.to_csv(out_csv_file)      


def process_sample_code15(exam):
    record_path = os.path.join(args.data_folder, str(exam[1]['trace_file']))
    exam_id = exam[1]['exam_id']
    out_path = os.path.join(args.output_folder, str(exam[1]['exam_id']))
    resample_and_save_record_hdf5(record_path, exam_id, 360, out_path, nk_clean=args.nk_clean)


if __name__ == '__main__':
    # make directory for the output
    clean_and_create_directory(args.output_folder)

    if args.dataset == 'mimic':
        records = pd.read_csv(os.path.join(args.data_folder, 'machine_measurements.csv'))
        Parallel(n_jobs=-1)(delayed(process_sample_mimic)(sample) for i, sample in tqdm(records.iterrows()))
        process_csv_file_mimic(args.label_file, os.path.join(args.output_folder, 'records_w_diag_icd10_labelled.csv'))

    if args.dataset == 'code15':
        exams = pd.read_csv(args.label_file)
        Parallel(n_jobs=-1)(delayed(process_sample_code15)(exam) for exam in tqdm(exams.iterrows()))
        process_csv_file_code15(args.label_file, os.path.join(args.output_folder, 'exams_labelled.csv'))


