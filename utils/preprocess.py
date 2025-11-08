import os
import numpy as np
import scipy
import scipy.io.wavfile as wav
import torch
from scipy.signal import butter, filtfilt, lfilter, resample
from math import ceil
import pandas as pd


def butterworth_filter(data, sampling_rate, cutoff_freq, btype, order):

    nyquist_freq = sampling_rate / 2.
    cutoff_freq = cutoff_freq / nyquist_freq


    b, a = butter(N=order, Wn=cutoff_freq, btype=btype)


    filtered_data = lfilter(b, a, data)

    return filtered_data


def process_lung(database_path, output_path, fc_lung, train: bool, classify: bool):
    if train:
        output_path = os.path.join(output_path, 'lung', 'train')
        classify = False
    else:
        output_path = os.path.join(output_path, 'lung', 'test')

    if classify:
        if not os.path.exists(os.path.join(output_path, f'{fc_lung}Hz', 'crack')):
            os.makedirs(os.path.join(output_path, f'{fc_lung}Hz', 'crack'))
        if not os.path.exists(os.path.join(output_path, f'{fc_lung}Hz', 'wheeze')):
            os.makedirs(os.path.join(output_path, f'{fc_lung}Hz', 'wheeze'))
        if not os.path.exists(os.path.join(output_path, f'{fc_lung}Hz', 'none')):
            os.makedirs(os.path.join(output_path, f'{fc_lung}Hz', 'none'))
    else:
        if not os.path.exists(os.path.join(output_path, f'{fc_lung}Hz')):
            os.makedirs(os.path.join(output_path, f'{fc_lung}Hz'))

    fs = 4000
    window_length = 200  #
    overlap = 150  #
    segment_length = 512 * (window_length - overlap) + overlap  # 512 frames

    count = 0
    for file in os.listdir(database_path):

        if file.endswith('.wav'):
            print(count)
            count = count + 1
            wav_path = os.path.join(database_path, file)
            txt_path = os.path.join(database_path, file.replace('.wav', '.txt'))

            # read files
            sample_rate, data = wav.read(wav_path)

            # filter
            data = butterworth_filter(data, sample_rate, 1200, btype='low', order=6)
            data = butterworth_filter(data, sample_rate, fc_lung, btype='high', order=6)
            data = resample(data, int(len(data) * fs / sample_rate))
            data = (data - np.average(data)) / np.std(data)
            length_data = len(data)


            if length_data < segment_length:
                continue

            num_segment = ceil(length_data / segment_length)
            starts_portion = np.linspace(0, 1, num=num_segment, endpoint=True)
            starts = ((length_data - segment_length) * starts_portion).astype(int)

            # 读取对应的txt文件
            with open(txt_path, 'r') as f:
                annotations = [list(map(float, line.strip().split())) for line in f.readlines()]

            if classify:
                # 创建标签数组
                labels = np.zeros((length_data,), dtype=int)  # 0: none, 1: crack, 2: wheeze
                for start_time, end_time, crack, wheeze in annotations:
                    start_sample = int(start_time * fs)
                    end_sample = int(end_time * fs)
                    if crack == 1:
                        labels[start_sample:end_sample] |= 1  # 标记crack
                    if wheeze == 1:
                        labels[start_sample:end_sample] |= 2  # 标记wheeze

            # clip according to the labels and save
            for i, start in enumerate(starts):
                segment = data[start:start + segment_length]
                f, t, Z = scipy.signal.stft(segment, fs=fs, boundary=None,
                                            nperseg=window_length, noverlap=overlap, padded=False)

                Z_tensor = torch.tensor(Z, dtype=torch.complex64)

                if classify:
                    # classification and save
                    segment_labels = labels[start:start + segment_length]
                    has_crack = np.any(segment_labels == 1)
                    has_wheeze = np.any(segment_labels == 2)

                    if has_crack:
                        torch.save(Z_tensor, os.path.join(output_path, f'{fc_lung}Hz', 'crack', f'{file}_{i}.pt'))
                    if has_wheeze:
                        torch.save(Z_tensor, os.path.join(output_path, f'{fc_lung}Hz', 'wheeze', f'{file}_{i}.pt'))
                    if not has_crack and not has_wheeze:
                        torch.save(Z_tensor, os.path.join(output_path, f'{fc_lung}Hz', 'none', f'{file}_{i}.pt'))
                else:
                    torch.save(Z_tensor, os.path.join(output_path, f'{fc_lung}Hz', f'{file}_{i}.pt'))


def process_heart(database_path, output_path, fc_heart, train: bool, classify: bool):
    if train:
        csv_path = os.path.join(database_path, "data_train.csv")
        output_path = os.path.join(output_path, 'heart', 'train')
        classify = False
    else:
        output_path = os.path.join(output_path, 'heart', 'test')
        csv_path = os.path.join(database_path, "data_test.csv")


    output_dirs = {
        "Benign": "Benign",
        "MR": "MR",
        "AS": "AS",
        "Controls": "Normal",
        "Normal ": "Normal"
    }

    fs = 4000
    window_length = 200  #
    overlap = 150  #
    segment_length = 512 * (window_length - overlap) + overlap  # 512 frames

    # create the output folders
    if classify:
        for folder in output_dirs.values():
            if not os.path.exists(os.path.join(output_path, f'{fc_heart}Hz', folder)):
                os.makedirs(os.path.join(output_path, f'{fc_heart}Hz', folder))
    else:
        if not os.path.exists(os.path.join(output_path, f'{fc_heart}Hz')):
            os.makedirs(os.path.join(output_path, f'{fc_heart}Hz'))

    # read CSV
    df = pd.read_csv(csv_path, usecols=[0, 3], names=["filename", "label"], header=0)

    # handle each WAV file
    count = 0
    for idx, row in df.iterrows():
        print(count)
        count += 1
        wav_path = os.path.join(database_path, f"{row['filename']}.wav")
        if not os.path.exists(wav_path):
            print(f"Warning: {wav_path} not found.")
            continue

        # read wav data
        sample_rate, data = wav.read(wav_path)
        if data.ndim > 1:  # to single channel
            data = np.mean(data, axis=1)

        # filter
        data = butterworth_filter(data, sample_rate, fc_heart, btype='low', order=6)
        # data = butterworth_filter(data, sample_rate, 10, btype='high', order=2)
        data = resample(data, int(len(data) * fs / sample_rate))
        data = (data - np.average(data)) / np.std(data)

        # segment
        length_data = len(data)
        if length_data < segment_length:
            continue
        num_segment = ceil(length_data / segment_length)
        starts_portion = np.linspace(0, 1, num=num_segment, endpoint=True)
        starts = ((length_data - segment_length) * starts_portion).astype(int)

        # save
        label = row['label']
        if train:
            subset = row['filename'][0]
            if subset != 'a' or label not in output_dirs:
                for i, start in enumerate(starts):
                    segment = data[start: start + segment_length]
                    f, t, Z = scipy.signal.stft(segment, fs=fs, boundary=None,
                                                nperseg=window_length, noverlap=overlap, padded=False)
                    Z_tensor = torch.tensor(Z, dtype=torch.complex64)
                    save_path = os.path.join(output_path, f'{fc_heart}Hz', f"{row['filename']}_{i}.pt")
                    torch.save(Z_tensor, save_path)

        elif label in output_dirs:
            if classify:
                for i, start in enumerate(starts):
                    segment = data[start: start + segment_length]
                    f, t, Z = scipy.signal.stft(segment, fs=fs, boundary=None,
                                                nperseg=window_length, noverlap=overlap, padded=False)

                    Z_tensor = torch.tensor(Z, dtype=torch.complex64)
                    save_path = os.path.join(output_path, f'{fc_heart}Hz',
                                             output_dirs[label], f"{row['filename']}_{i}.pt")
                    torch.save(Z_tensor, save_path)

            else:
                for i, start in enumerate(starts):
                    segment = data[start: start + segment_length]
                    f, t, Z = scipy.signal.stft(segment, fs=fs, boundary=None,
                                                nperseg=window_length, noverlap=overlap, padded=False)

                    Z_tensor = torch.tensor(Z, dtype=torch.complex64)
                    save_path = os.path.join(output_path, f'{fc_heart}Hz', f"{row['filename']}_{i}.pt")
                    torch.save(Z_tensor, save_path)
        else:
            print(f"Warning: {label} not found.")



def preprocess(database_path, output_path, signal_type, fc, train, classify):
    if signal_type == 'heart':
        if fc is None:
            fc = 250
        process_heart(database_path=os.path.join(database_path, 'heart_storage'),
                  output_path=output_path,
                  fc_heart=fc, train=train, classify=classify)
    elif signal_type == 'lung' and train == True:
        if fc is None:
            fc = 60
        process_lung(database_path=os.path.join(database_path, 'lung_storage_train'),
                  output_path=output_path,
                  fc_lung=fc, train=train, classify=classify)
    elif signal_type == 'lung' and train == False:
        if fc is None:
            fc = 60
        process_lung(database_path=os.path.join(database_path, 'lung_storage_test'),
                  output_path=output_path,
                  fc_lung=fc, train=train, classify=classify)
    else:
        RuntimeError('No such signal_type')


