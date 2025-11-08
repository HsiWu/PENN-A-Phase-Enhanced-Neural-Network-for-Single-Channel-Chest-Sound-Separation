import os.path

import numpy as np
import torch


def calculate_mean_and_std(vector):

    valid_mask = np.isfinite(vector)



    valid_values = vector[valid_mask]

    if valid_values.size == 0:
        return "No valid values"


    mean_value = np.mean(valid_values)
    std_value = np.std(valid_values)

    return mean_value


def calculate(cut_off_heart, cut_off_lung, model, path,
              complex_mask: bool, pure_phase: bool, dual_decoder:bool, classify=None):
    if model == 'Dual_Trans':
        model = path + '_Trans'
    elif model == 'Dual_Trans_Decoder':
        model = path + '_Trans_Decoder'
    else:
        RuntimeError('The mode that you entered does not exist')

    if classify is None:
        a = torch.load(os.path.join('outcomes', 'outcome' + '-' + model + '-' + f'{complex_mask}' + '-' + f'{pure_phase}' +
                           '-' + f'{dual_decoder}' + '-' +
                       cut_off_heart + '-' + cut_off_lung ))
    else:
        a = torch.load(os.path.join('outcomes', 'outcome' + '-' + model + '-' + f'{complex_mask}' + '-' + f'{pure_phase}' +
                           '-' + f'{dual_decoder}' + '-' +
                       cut_off_heart + '-' + cut_off_lung  + '-' + classify))

    print(model + '-' + f'{complex_mask}' + '-' + f'{pure_phase}' + '-' +
          cut_off_heart + '-' + cut_off_lung)
    # pig = a['a']
    print('heart SI-SDRi', calculate_mean_and_std(a['a']), 'heart mSI-SDRi', calculate_mean_and_std(a['b']), 'heart pSI-SDRi', calculate_mean_and_std(a['c']))
    print('lung SI-SDRi', calculate_mean_and_std(a['d']), 'lung mSI-SDRi', calculate_mean_and_std(a['e']), 'lung pSI-SDRi', calculate_mean_and_std(a['f']))
    print()
