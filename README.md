
# A Phase-Enhanced Neural Network with Dual-Path Transformer for Single-Channel Chest Sound Separation

This repository contains the implementation of the paper **"A Phase-Enhanced Neural Network with Dual-Path Transformer for Single-Channel Chest Sound Separation"** published in the **IEEE Journal of Biomedical and Health Informatics**.

## Project Structure

```
PENN
├── data                # Folder storing raw and preprocessed data
├── models              # Folder storing the trained models
├── outcomes            # Folder storing training/testing losses and results
├── random_pairs        # Folder storing fixed test sample lists
├── utils               # Folder storing utility scripts
├── main_outcome.py     # Script to output the test results
├── main_preprocess.py  # Script for data preprocessing
├── main_test.py        # Script for testing the model
├── main_train.py       # Script for training the model
└── requirements.txt    # Required dependencies
```

The .zip files should be unzipped in the project folder.
Additionally, the data folder should be downloaded from the "Releases" section and also unzipped within the project folder.

## Requirements

To run the project, make sure you have the following dependencies installed. You can install them using `pip`:

```
torch==2.1.2+cu118
torchaudio==2.1.2+cu118
torchvision==0.16.2+cu118
numpy==1.26.4
scipy==1.13.1
fast-bss-eval==0.1.4
pandas==2.2.3
```

You can install them by running:

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Data Preprocessing

First, preprocess the data using the `main_preprocess.py` script. It requires two parameters: `signal_type` and `train`.

#### Usage:

```bash
python main_preprocess.py --signal_type 'heart' --train True
python main_preprocess.py --signal_type 'lung' --train True
python main_preprocess.py --signal_type 'heart' --train False
python main_preprocess.py --signal_type 'lung' --train False
```

- `signal_type`: Choose between `'heart'` and `'lung'` signals.
- `train`: Set to `True` for training data and `False` for testing data.

### 2. Training the Model

Once the data preprocessing is done, you can train the model using the `main_train.py` script.

#### Usage:

```bash
python main_train.py
```

### 3. Testing the Model

After training the model, run the `main_test.py` script to test it.

#### Usage:

```bash
python main_test.py
```

### 4. Output Test Results

Finally, run the `main_outcome.py` script to output the test results.

#### Usage:

```bash
python main_outcome.py
```

### 5. Additional Parameters

You can specify additional parameters in the scripts. For example, if you want to train and test the model with a complex-valued mask, use the following commands:

```bash
python main_train.py --complex_mask True
python main_test.py --complex_mask True
python main_outcome.py --complex_mask True
```

For a full list of available parameters, please refer to the code in the respective scripts.

## Citation

If you use this code in your research, please cite the corresponding paper.

