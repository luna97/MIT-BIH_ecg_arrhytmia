## MIMIC-IV

To preprocess the dataset and downsample the recordings it to a frequency of 360 use the following script:

```bash
python prepare_dataset.py --data_folder /home/datasets/mimic-iv-ecg/raw/ \ 
    --nk_clean \
    --dataset mimic \
    --output_folder /home/datasets/mimic-iv-ecg/360hz_nkclean/ \
    --label_file /home/datasets/mimic-iv-ecg/raw/records_w_diag_icd10.csv
```

The data folder argument should contain the raw data downloaded from: https://physionet.org/content/mimic-iv-ecg/1.0/files/p1000/p10000032/s40689238/#files-panel

I suggest to use the aws cli to download it:

```bash
aws s3 sync --no-sign-request s3://physionet-open/mimic-iv-ecg/1.0/ <DESTINATION>
```

Where `<DESTINATION>` will be then your `--data_folder` argument.

Use the argument `--nk_clean` to apply a 0.5 Hz high-pass butterworth filter, followed by powerline filtering.
See https://neuropsychology.github.io/NeuroKit/functions/ecg.html#neurokit2.ecg.ecg_clean for more information about preprocessing

With the argument `--dataset` you specify to which dataset apply preprocessing, in this case use `mimic`


## CODE15


## MIT-BIH arrhythmia
