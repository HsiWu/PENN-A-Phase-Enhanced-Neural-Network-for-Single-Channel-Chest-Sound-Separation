import numpy as np
import torch
import utils.Module as Module
import utils.MyData_test as MyData_test
import utils.test_metrics as test_metrics
from torch.utils.data import DataLoader
import os


def test(root_dir, cut_off_heart, cut_off_lung, model, path, complex_mask: bool, pure_phase: bool,
         dual_decoder:bool, d_model: int = 64,
         num_layers: int = 3, nhead: int = 16,
         length_time: int = 256, length_fre: int = 51, test_num: int = 5000, classify=None):
    batch_size = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cut_off_lung != '40Hz' or cut_off_lung != '60Hz' or cut_off_lung != '80Hz' or cut_off_lung != '100Hz' \
            or cut_off_heart != '200Hz' or cut_off_heart != '225Hz' \
            or cut_off_heart != '250Hz' or cut_off_heart != '275Hz':
        RuntimeError('The cut-off frequency you entered does not exist')

    if model == 'Dual_Trans':
        the_model = Module.DualTrans(int(1 * d_model), num_layers, nhead, int(1 * d_model) * 2, path, device=device,
                                     complex_mask=complex_mask, pure_phase=pure_phase).to(device)
        model = path + '_Trans'
    elif model == 'Dual_Trans_Decoder':
        the_model = Module.DualTrans_Decoder(d_model, num_layers, nhead, d_model * 2, path, device=device,
                                             dual_decoder=dual_decoder,
                                     complex_mask=complex_mask, pure_phase=pure_phase).to(device)
        model = path + '_Trans_Decoder'
    else:
        RuntimeError('The mode that you entered does not exist')

    print(model + '-' + f'{complex_mask}' + '-' + f'{pure_phase}' +
                           '-' + f'{dual_decoder}' + '-' + cut_off_heart + '-' + cut_off_lung)

    num_params = sum(p.numel() for p in the_model.parameters() if p.requires_grad)
    print(num_params)
    model_stored = torch.load(os.path.join('models', 'model' + '-' + model + '-' + f'{complex_mask}' + '-' + f'{pure_phase}' +
                           '-' + f'{dual_decoder}' + '-' + cut_off_heart + '-' + cut_off_lung), map_location=device)
    the_model.load_state_dict(model_stored['model_state_dict'])

    # cut_off_heart = 'own'
    if classify is None:
        dataset_test = MyData_test.My_Data(root_dir, os.path.join('heart', 'test', cut_off_heart),
                                           os.path.join('lung', 'test', cut_off_lung),
                                           complex_mask=complex_mask, steps=test_num, classify=classify)
    elif classify == 'AS' or classify == 'Benign' or classify == 'MR' or classify == 'Normal':
        dataset_test = MyData_test.My_Data(root_dir, os.path.join('heart', 'test', cut_off_heart, classify),
                                           os.path.join('lung', 'test', cut_off_lung),
                                           complex_mask=complex_mask, steps=test_num, classify=classify)
    elif classify == 'crack' or classify == 'none' or classify == 'wheeze':
        dataset_test = MyData_test.My_Data(root_dir, os.path.join('heart', 'test', cut_off_heart),
                                           os.path.join('lung', 'test', cut_off_lung, classify),
                                           complex_mask=complex_mask, steps=test_num, classify=classify)
    else:
        dataset_test = 0
        RuntimeError('None such error')
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    mask_np = np.ones((length_time, length_time), dtype=bool)
    for i in range(length_time):
        for j in range(length_time):
            if abs(i - j) <= length_time // 2:
                mask_np[i][j] = False
    mask_time = torch.tensor(data=mask_np, dtype=torch.bool).to(device)

    mask_np = np.ones((length_fre, length_fre), dtype=bool)
    for i in range(length_fre):
        for j in range(length_fre):
            if abs(i - j) <= length_fre:
                mask_np[i][j] = False
    mask_fre = torch.tensor(data=mask_np, dtype=torch.bool).to(device)
    # the_pool = nn.AvgPool2d((2, 2), padding=(0, 0))

    count = 0

    heart_si_sdri = np.zeros((test_num, 1))
    lung_si_sdri = np.zeros((test_num, 1))
    heart_msi_sdri = np.zeros((test_num, 1))
    lung_msi_sdri = np.zeros((test_num, 1))
    heart_psi_sdri = np.zeros((test_num, 1))
    lung_psi_sdri = np.zeros((test_num, 1))

    for i, data_test in enumerate(dataloader_test):

        if complex_mask:
            heart_real, heart_imag, lung_real, lung_imag, \
            mask_real_heart, mask_imag_heart, mask_real_lung, mask_imag_lung, \
            mix_real, mix_imag = data_test
            # mask_signals = torch.concatenate((mask_real_heart, mask_imag_heart, mask_real_lung, mask_imag_lung),
            #                                  dim=0)
        else:
            heart_real, heart_imag, lung_real, lung_imag, \
            mask_heart, mask_lung, mix_real, mix_imag = data_test

            # mask_signals = torch.concatenate((mask_heart, mask_lung), dim=0)

        mix_real = mix_real.to(torch.float).to(device)
        mix_imag = mix_imag.to(torch.float).to(device)
        # mask_signals = mask_signals.to(torch.float).to(device)
        heart_real = heart_real.to(torch.float).to(device)
        heart_imag = heart_imag.to(torch.float).to(device)
        lung_real = lung_real.to(torch.float).to(device)
        lung_imag = lung_imag.to(torch.float).to(device)
        mix = torch.concatenate((mix_real, mix_imag), dim=0)
        # target = torch.concatenate((heart_real, heart_imag, lung_real, lung_imag), dim=0)



        with torch.no_grad():
            the_model.eval()
            mask_esti, signal_esti = the_model(src=mix[:, :51, :], mask_time=mask_time, mask_fre=mask_fre)



        mask_esti = mask_esti.to('cpu').numpy()
        heart_real = torch.squeeze(heart_real).to('cpu').numpy()
        heart_imag = torch.squeeze(heart_imag).to('cpu').numpy()
        lung_real = torch.squeeze(lung_real).to('cpu').numpy()
        lung_imag = torch.squeeze(lung_imag).to('cpu').numpy()
        mix_real = torch.squeeze(mix_real).to('cpu').numpy()
        mix_imag = torch.squeeze(mix_imag).to('cpu').numpy()
        signal_esti = signal_esti.to('cpu').numpy()

        heart_target_stft = heart_real + 1j * heart_imag
        lung_target_stft = lung_real + 1j * lung_imag
        mix_stft = mix_real + 1j * mix_imag

        if model == 'Dual_Trans' or model == 'Time_Trans' or model == 'Fre_Trans':

            if complex_mask:
                K = 10
                mask_esti = - np.log((K - mask_esti) / (K + mask_esti))
                mask_heart = mask_esti[0, :, :] + 1j * mask_esti[1, :, :]
                mask_lung = mask_esti[2, :, :] + 1j * mask_esti[3, :, :]
                heart_esti_stft = 10 * mix_stft[:51, :]  * mask_heart
                lung_esti_stft = 10 * mix_stft[:51, :]  * mask_lung

            else:
                heart_esti_stft = mix_stft[:51, :] * mask_esti[0, :, :]
                lung_esti_stft = mix_stft[:51, :]  * mask_esti[1, :, :]
        else:
            heart_esti_stft = signal_esti[0, :, :] + 1j * signal_esti[1, :, :]
            lung_esti_stft = signal_esti[2, :, :] + 1j * signal_esti[3, :, :]

        heart_esti_stft = np.concatenate((heart_esti_stft, np.zeros((50, 512), dtype=np.complex64)), axis=0)
        lung_esti_stft = np.concatenate((lung_esti_stft, mix_stft[51:101, :]), axis=0)

        si_sdr, si_sdr_naive, msi_sdr, psi_sdr = test_metrics.metrics(heart_esti_stft, lung_esti_stft, heart_target_stft, lung_target_stft, mix_stft,
                                                                      wav=False, fs=4000, window_length=200, overlap=150)


        heart_si_sdri[count] = si_sdr[0] - si_sdr_naive[0]
        lung_si_sdri[count] = si_sdr[1] - si_sdr_naive[1]
        heart_msi_sdri[count] = 0.5 * (msi_sdr[0] - si_sdr_naive[0]) + 0.5 * (si_sdr[0] - psi_sdr[0])
        lung_msi_sdri[count] = 0.5 * (msi_sdr[1] - si_sdr_naive[1]) + 0.5 * (si_sdr[1] - psi_sdr[1])
        heart_psi_sdri[count] = 0.5 * (psi_sdr[0] - si_sdr_naive[0]) + 0.5 * (si_sdr[0] - msi_sdr[0])
        lung_psi_sdri[count] = 0.5 * (psi_sdr[1] - si_sdr_naive[1]) + 0.5 * (si_sdr[1] - msi_sdr[1])

        if count % 500 == 0:
            print(count)
        count += 1

    print('heart SI-SDRi', np.average(heart_si_sdri), 'lung SI-SDRi', np.average(lung_si_sdri))
    # print(np.std(heart_si_sdri), np.std(lung_si_sdri))
    print('heart mSI-SDRi', np.average(heart_msi_sdri), 'lung mSI-SDRi', np.average(lung_msi_sdri))
    # print(np.std(heart_si_sdri), np.std(lung_si_sdri))
    print('heart pSI-SDRi', np.average(heart_psi_sdri), 'lung pSI-SDRi', np.average(lung_psi_sdri))
    # print(np.std(heart_si_sdri), np.std(lung_si_sdri))

    if classify:
        torch.save({'a': heart_si_sdri, 'b': heart_msi_sdri, 'c': heart_psi_sdri,
                    'd': lung_si_sdri, 'e': lung_msi_sdri, 'f': lung_psi_sdri},
                   os.path.join('outcomes', 'outcome' + '-' + model + '-' + f'{complex_mask}' + '-' + f'{pure_phase}' +
                           '-' + f'{dual_decoder}' + '-' +
                   cut_off_heart + '-' + cut_off_lung + '-' + classify))
    else:
        torch.save({'a': heart_si_sdri, 'b': heart_msi_sdri, 'c': heart_psi_sdri,
                    'd': lung_si_sdri, 'e': lung_msi_sdri, 'f': lung_psi_sdri},
                   os.path.join('outcomes', 'outcome' + '-' + model + '-' + f'{complex_mask}' + '-' + f'{pure_phase}' +
                           '-' + f'{dual_decoder}' + '-' +
                   cut_off_heart + '-' + cut_off_lung))
