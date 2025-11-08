import numpy as np
import scipy
import fast_bss_eval


def metrics(heart_esti, lung_esti, heart_target, lung_target, mix, wav: bool,
            fs: int = 4000, window_length: int = 200, overlap: int = 150):
    if wav:
        t, f, heart_esti_stft = scipy.signal.stft(heart_esti, fs=fs, boundary=None,
                                                  nperseg=window_length, noverlap=overlap, padded=False)
        t, f, lung_esti_stft = scipy.signal.stft(lung_esti, fs=fs, boundary=None,
                                                 nperseg=window_length, noverlap=overlap, padded=False)
        t, f, heart_target_stft = scipy.signal.stft(heart_target, fs=fs, boundary=None,
                                                    nperseg=window_length, noverlap=overlap, padded=False)
        t, f, lung_target_stft = scipy.signal.stft(lung_target, fs=fs, boundary=None,
                                                   nperseg=window_length, noverlap=overlap, padded=False)
        t, f, mix_stft = scipy.signal.stft(mix, fs=fs, boundary=None,
                                           nperseg=window_length, noverlap=overlap, padded=False)
    else:
        heart_esti_stft = heart_esti
        lung_esti_stft = lung_esti
        heart_target_stft = heart_target
        lung_target_stft = lung_target
        mix_stft = mix

    heart_esti_stft_phase = np.angle(heart_esti_stft)
    lung_esti_stft_phase = np.angle(lung_esti_stft)
    mix_stft_phase = np.angle(mix_stft)

    heart_only_mag_esti_stft = np.abs(heart_esti_stft) * np.exp(1j * mix_stft_phase)
    heart_only_phase_esti_stft = np.abs(mix_stft) * np.exp(1j * heart_esti_stft_phase)
    lung_only_mag_esti_stft = np.abs(lung_esti_stft) * np.exp(1j * mix_stft_phase)
    lung_only_phase_esti_stft = np.abs(mix_stft) * np.exp(1j * lung_esti_stft_phase)

    heart_esti = scipy.signal.istft(heart_esti_stft, fs=fs, boundary='zeros',
                                    nperseg=window_length, noverlap=overlap)
    lung_esti = scipy.signal.istft(lung_esti_stft, fs=fs, boundary='zeros',
                                   nperseg=window_length, noverlap=overlap)
    heart_only_mag_esti = scipy.signal.istft(heart_only_mag_esti_stft, fs=fs, boundary='zeros',
                                             nperseg=window_length, noverlap=overlap)
    heart_only_phase_esti = scipy.signal.istft(heart_only_phase_esti_stft, fs=fs, boundary='zeros',
                                               nperseg=window_length, noverlap=overlap)
    lung_only_mag_esti = scipy.signal.istft(lung_only_mag_esti_stft, fs=fs, boundary='zeros',
                                            nperseg=window_length, noverlap=overlap)
    lung_only_phase_esti = scipy.signal.istft(lung_only_phase_esti_stft, fs=fs, boundary='zeros',
                                              nperseg=window_length, noverlap=overlap)
    heart_target = scipy.signal.istft(heart_target_stft, fs=fs, boundary='zeros',
                                      nperseg=window_length, noverlap=overlap)
    lung_target = scipy.signal.istft(lung_target_stft, fs=fs, boundary='zeros',
                                     nperseg=window_length, noverlap=overlap)
    mix_target = scipy.signal.istft(mix_stft, fs=fs, boundary='zeros',
                                    nperseg=window_length, noverlap=overlap)

    ref = np.concatenate((heart_target[1][:, np.newaxis], lung_target[1][:, np.newaxis]), axis=1)
    est = np.concatenate((heart_esti[1][:, np.newaxis], lung_esti[1][:, np.newaxis]), axis=1)
    est_only_mag = np.concatenate((heart_only_mag_esti[1][:, np.newaxis], lung_only_mag_esti[1][:, np.newaxis]),
                                  axis=1)
    est_only_phase = np.concatenate(
        (heart_only_phase_esti[1][:, np.newaxis], lung_only_phase_esti[1][:, np.newaxis]), axis=1)

    est_naive = np.concatenate((mix_target[1][:, np.newaxis], mix_target[1][:, np.newaxis]), axis=1)

    error1 = np.sum(np.abs(heart_only_phase_esti[1] - mix_target[1]))
    error2 = np.sum(np.abs(lung_only_phase_esti[1] - mix_target[1]))
    error3 = np.sum(np.abs(heart_only_phase_esti_stft - mix_stft))
    error4 = np.sum(np.abs(lung_only_phase_esti_stft - mix_stft))
    error5 = np.sum(np.abs(heart_esti_stft_phase - mix_stft_phase))
    error6 = np.sum(np.abs(lung_esti_stft_phase - mix_stft_phase))

    si_sdr, si_sir, si_sar, si_perm = \
        fast_bss_eval.si_bss_eval_sources(ref.T, est.T, clamp_db=80)
    si_sdr_naive, si_sir_naive, si_sar_naive, si_perm_naive = \
        fast_bss_eval.si_bss_eval_sources(ref.T, est_naive.T, clamp_db=80)
    msi_sdr, msi_sir, msi_sar, msi_perm = \
        fast_bss_eval.si_bss_eval_sources(ref.T, est_only_mag.T, clamp_db=80)
    psi_sdr, psi_sir, psi_sar, psi_perm = \
        fast_bss_eval.si_bss_eval_sources(ref.T, est_only_phase.T, clamp_db=80)

    si_sdr = si_sdr[np.argsort(si_perm)]
    si_sdr_naive = si_sdr_naive[np.argsort(si_perm)]
    msi_sdr = msi_sdr[np.argsort(si_perm)]
    psi_sdr = psi_sdr[np.argsort(si_perm)]

    return si_sdr, si_sdr_naive, msi_sdr, psi_sdr
