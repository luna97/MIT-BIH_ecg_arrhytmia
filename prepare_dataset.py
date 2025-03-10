import neurokit2 as nk
from tqdm import tqdm
import os
import wfdb
import argparse
import numpy as np
import shutil
import pandas as pd
from collections import Counter
from dataset.dataset_preparation_utils import resample_and_save_record_wfdb, resample_and_save_record_hdf5, clean_and_create_directory
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='Create dataset for MIT-BIH')
parser.add_argument('--data_folder', type=str, default='/media/Volume/data/MIMIC_IV/', help='Path to raw data folder')
parser.add_argument('--nk_clean', action='store_true', help='Use NeuroKit2 to clean the data')
parser.add_argument('--output_folder', type=str, required=True, help='the name of the output directory')
parser.add_argument('--dataset', type=str, required=True, help='the name of the dataset: mimic or code 15')
args = parser.parse_args()

def process_sample_mimic(sample):
    record_path = os.path.join(args.data_folder, 'files', 'p' + str(sample['subject_id'])[:4], 'p' + str(sample['subject_id']), 's' + str(sample['study_id']), str(sample['study_id']))
    out_path = os.path.join(args.output_folder, str(sample['study_id']))
    resample_and_save_record_wfdb(record_path, 360, out_path, nk_clean=args.nk_clean)

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

    if args.dataset == 'code15':
        exams = pd.read_csv(os.path.join(args.data_folder, 'exams.csv'))
        Parallel(n_jobs=-1)(delayed(process_sample_code15)(exam) for exam in tqdm(exams.iterrows()))


