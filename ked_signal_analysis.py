import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb
import os
from tqdm import tqdm

def signal_analyse(signal, sampling_rate=500, lead_index=1):
    signal = np.asarray(signal)
    if signal.ndim == 2:
        signal = signal[:, lead_index]  # 默认分析第2导联
    signal[np.isnan(signal)] = 0
    signal[np.isinf(signal)] = 0
    try:
        ecg_cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)
        _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate, method="neurokit")
        signal_rates = nk.signal_rate(rpeaks, sampling_rate=sampling_rate)
        _, waves = nk.ecg_delineate(ecg_cleaned, rpeaks, sampling_rate=sampling_rate, method="dwt", show=False,
                                    show_type='all')
    except:
        return 0, 0, 0, 0, 0, 0, 0, 0
    qrs = []
    rr = []
    pr = []
    qt = []
    pp = []
    rp = []
    tp = []
    for index, _ in enumerate(rpeaks['ECG_R_Peaks']):
        if index < len(waves['ECG_Q_Peaks']):
            P_R_interval = float(waves['ECG_Q_Peaks'][index] - waves['ECG_P_Onsets'][index]) / sampling_rate
            QRS_complex = float(waves['ECG_S_Peaks'][index] - waves['ECG_Q_Peaks'][index]) / sampling_rate
            qrs.append(QRS_complex)
            pr.append(P_R_interval)
        Q_T_interval = float(waves['ECG_T_Offsets'][index] - waves['ECG_R_Onsets'][index]) / sampling_rate
        if index == 0:
            RR_interval = None
        else:
            RR_interval = float(rpeaks['ECG_R_Peaks'][index] - rpeaks['ECG_R_Peaks'][index - 1]) / sampling_rate
            rr.append(RR_interval)

        qt.append(Q_T_interval)
        pp.append(waves['ECG_P_Peaks'][index]/1000)
        rp.append(rpeaks['ECG_R_Peaks'][index]/1000)
        tp.append(waves['ECG_T_Peaks'][index]/1000)

    dropna = lambda x: np.mean(np.array(x)[~np.isnan(np.array(x))]) if len(np.array(x)[~np.isnan(np.array(x))]) > 0 else 0
    int_dropna = lambda x: int(dropna(x)*1000)
    return int_dropna(rr), int_dropna(pr), int_dropna(qrs), int_dropna(qt), int((dropna(qt) / np.sqrt(dropna(rr)))*1000), int_dropna(pp), int_dropna(rp), int_dropna(tp)

def calculate_waveforms(signal_list, sampling_rate=500, lead_index=1):
        data_dict = {
        'RR_Interval': [],
        'PR_Interval': [],
        'QRS_Complex': [],
        'QT_Interval': [],
        'QTc_Interval': [],
        'P_Wave_Peak': [],
        'R_Wave_Peak': [],
        'T_Wave_Peak': []
    }

    for signal in tqdm(signal_list):
        rr, pr, qrs, qt, qtc, pp, rp, tp = signal_analyse(signal, sampling_rate=sampling_rate, lead_index=lead_index)
        data_dict['RR_Interval'].append(rr)
        data_dict['PR_Interval'].append(pr)
        data_dict['QRS_Complex'].append(qrs)
        data_dict['QT_Interval'].append(qt)
        data_dict['QTc_Interval'].append(qtc)
        data_dict['P_Wave_Peak'].append(pp)
        data_dict['R_Wave_Peak'].append(rp)
        data_dict['T_Wave_Peak'].append(tp)
    return data_dict