import pandas as pd
import numpy as np
import multiprocessing
from itertools import repeat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import ast
import pickle
import csv
import json
import requests
import os
import wfdb
from tqdm import tqdm

from sklearn.model_selection import train_test_split

"""
This data processing document has been modified from this project: https://github.com/helme/ecg_ptbxl_benchmarking
"""

def handler_data(experiment_name, task, datafolder, sampling_frequency = 500, min_samples = 0, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    #加载原始信号和标签
    data, raw_labels = load_dataset(datafolder, sampling_frequency)

    #加入英文报告
    report_transtion = pd.read_csv('/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/ptb-xl/report_translation_final.csv')
    raw_labels['translation_report'] = report_transtion['target']

    #加入波形特征
    # 定义波形特征描述函数
    def get_wave_info(data):
        text_describe = ""
        text_describe += f" RR: {data['RR_Interval']}"
        text_describe += f" PR: {data['PR_Interval']}"
        text_describe += f" QRS: {data['QRS_Complex']}"
        text_describe += f" QT/QTc: {data['QT_Interval']}/{data['QTc_Interval']}"
        text_describe += f" P/R/T Wave: {data['P_Wave_Peak']}/{data['R_Wave_Peak']}/{data['T_Wave_Peak']}"
        return text_describe

    # 新建 translation_report_with_wave 列，拼接报告和波形特征
    def append_wave_to_report(row):
        return str(row['translation_report']) + get_wave_info(row)

    raw_labels['translation_report_with_wave'] = raw_labels.apply(append_wave_to_report, axis=1)

    # Preprocess label data：(21799,29)标签聚合 
    labels = compute_label_aggregations(raw_labels, datafolder, task)

    # Select relevant data and convert to one-hot (21799,1000,12), (21799,30)   Y:(21388, 71)
    #根据task筛选出有效数据，并将多标签编码(已去除)
    data, labels = select_data(data, labels, task, min_samples,
                                                          './' + experiment_name + '/data/')
    input_shape = data[0].shape #信号
    print(input_shape)
    print(labels.columns)
    #print(labels['scp_codes_len', 'all_scp', 'all_scp_len'][:5])

    #对信号独立z-score归一化
    def zscore_norm(ecg_signal):
        # (channels, timesteps)
        mean = np.mean(ecg_signal, axis=1, keepdims=True)
        std = np.std(ecg_signal, axis=1, keepdims=True)
        ecg_norm=(ecg_signal - mean) / (std + 1e-8)
        return ecg_norm

    ecg_signal_norm = [zscore_norm(sig) for sig in data]


    # 随机划分数据集（8:1:1）X信号 Y报告与标签 
    X_temp, X_test, y_temp, y_test = train_test_split( #划分出测试集
        ecg_signal_norm, labels, test_size=test_size, random_state=random_state, stratify=None
    )   
    val_ratio = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split( #划分出训练与验证集
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=None
    )

    # 保存信号
    X_train.dump(f'./{experiment_name}/data/signal_train_norm.npy')
    X_val.dump(f'./{experiment_name}/data/signal_val_norm.npy')
    X_test.dump(f'./{experiment_name}/data/signal_test_norm.npy')

    # 保存标签（如diagnostic）、报告（如report）分开
    label_cols = ['superdiagnostic']    # 换成你的标签列
    report_col = ['translation_report_with_wave']        # 换成你的报告列
    #标签
    label_train[label_cols].to_csv(f'./{experiment_name}/data/label_train.csv', index=True)
    label_val[label_cols].to_csv(f'./{experiment_name}/data/label_val.csv', index=True)
    label_test[label_cols].to_csv(f'./{experiment_name}/data/label_test.csv', index=True)
    #报告
    report_train[report_col].to_csv(f'./{experiment_name}/data/report_train.csv', index=True)
    report_val[report_col].to_csv(f'./{experiment_name}/data/report_val.csv', index=True)
    report_test[report_col].to_csv(f'./{experiment_name}/data/report_test.csv', index=True)

   
def load_raw_data_ptbxl(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])      # (21799, 1000, 12)
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    return data
def load_dataset(path, sampling_rate, release=False):
    if path.split('/')[-3] == 'ptb-xl':
        # load and convert annotation data
        Y = pd.read_csv(path+'new_ptbxl_database.csv', index_col='ecg_id')  # (21799, 27)
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data_ptbxl(Y, sampling_rate, path)
    else:
        raise ValueError("Invalid path or dataset type. Expected 'ptb-xl' in the path.")

    # elif path.split('/')[-2] == 'ICBEB':
    #     # load and convert annotation data
    #     Y = pd.read_csv(path+'icbeb_database.csv', index_col='ecg_id')
    #     Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    #
    #     # Load raw signal data
    #     X = load_raw_data_icbeb(Y, sampling_rate, path)

    return X, Y

def select_data(XX,YY, ctype, min_samples, outputfolder):
    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]

    
        # 筛选出diagnostic_len为0的index和report
        # res_X = XX[YY.diagnostic_len == 0]
        # res_Y = YY[YY.diagnostic_len == 0]
        # res_YY = res_Y[YY.strat_fold <= 8]
        # res_YY.to_csv("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1/data/res_labels.csv", index=True)
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
    
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
  
    elif ctype == 'form':
        # filter
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['form_len'] = YY.form.apply(lambda x: len(x))
        # select
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
   
    elif ctype == 'rhythm':
        # filter
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
        # select
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]

    elif ctype == 'all':
        # filter
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        # select
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]

    else:
        X, Y = XX, YY

    # save LabelBinarizer
 
    return X, Y

def compute_label_aggregations(df, folder, ctype):
#对每条记录的 scp_codes 字典做不同聚合处理（如diagnostic、form、rhythm等），生成新字段（列）。
    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(folder+'scp_statements.csv', index_col=0)

    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):    # 提取超类标签
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        if ctype == 'diagnostic':
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
    elif ctype == 'form':
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['form'] = df.scp_codes.apply(aggregate_form)
        df['form_len'] = df.form.apply(lambda x: len(x))
    elif ctype == 'rhythm':
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
    elif ctype == 'all':
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df
#信号标准化
def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp
#标准化信号
def preprocess_signals(X_train, X_validation, X_test, outputfolder):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))

    # Save Standardizer data
    with open(outputfolder + 'standard_scaler.pkl', 'wb') as ss_file:
        pickle.dump(ss, ss_file)

    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)
#报告翻译
def translate_report(report,csv_file_path):
    error_log_path = "/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/ptb-xl/error_log.txt"
    header = ['index', 'source', 'target']
    prompt_prefix_diagnosis = (
        "Help me translate the medical report from German into English. Please directly tell me the translation result, no other explanatory words. The origin medical report is: "
    )
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-822d9523f40b48028f1b4a340e2423ea",  # 替换为你的API密钥
    }

    # 1. 读取已完成index
    done_index = set()
    file_exists = os.path.exists(csv_file_path)
    if file_exists:
        with open(csv_file_path, 'r', newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    idx = int(row['index'])
                    if row['target'] and row['target'] != "error":
                        done_index.add(idx)
                except Exception:
                    continue

    # 2. 读取历史error_list
    if os.path.exists(error_log_path):
        with open(error_log_path, "r", encoding="utf-8") as ef:
            error_list = [line.strip() for line in ef if line.strip()]
    else:
        error_list = []

    # 3. 追加写模式打开，若新文件则写header
    with open(csv_file_path, 'a', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists or os.stat(csv_file_path).st_size == 0:
            writer.writerow(header)  # 只写一次header

        for idx, item in enumerate(report):
            if idx in done_index:
                continue  # 跳过已完成
            entry = [None] * len(header)
            entry[0] = idx
            entry[1] = item
            try:
                data = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": prompt_prefix_diagnosis + str(item)}
                    ]
                }
                json_data = json.dumps(data)
                response = requests.post(url=url, data=json_data, headers=headers)
                json_response = response.json()
                translated = json_response["choices"][0]["message"]["content"]
                print(idx, translated)
                entry[2] = translated.replace("\n\t", "").replace("\n", "")
                writer.writerow(entry)
            except Exception as e:
                error_list.append(str(item))
                entry[2] = "error"
                writer.writerow(entry)
                # 追加写入错误日志
                with open(error_log_path, "a", encoding="utf-8") as ef:
                    ef.write(str(item) + "\n")
    print(error_list)


def generate_ptb_label_gemini_augment():
    all_label_map = {'NDT': 'non-diagnostic T wave abnormalities',
                     'NST_': 'ST segment changes',
                     'DIG': 'digitalis-effect',
                     'LNGQT': 'long QT interval',
                     'NORM': 'normal ECG',
                     'IMI': 'inferior myocardial infarction',
                     'ASMI': 'anteroseptal myocardial infarction',
                     'LVH': 'left ventricular hypertrophy',
                     'LAFB': 'left anterior fascicular block',
                     'ISC_': 'myocardial ischemic',
                     'IRBBB': 'incomplete right bundle branch block',
                     '1AVB': 'first degree atrioventricular block',
                     'IVCD': 'intraventricular conduction disturbance (block)',
                     'ISCAL': 'anterolateral myocardial ischemic',
                     'CRBBB': 'complete right bundle branch block',
                     'CLBBB': 'complete left bundle branch block',
                     'ILMI': 'inferolateral myocardial infarction',
                     'LAO/LAE': 'left atrial overload/enlargement',
                     'AMI': 'anterior myocardial infarction',
                     'ALMI': 'anterolateral myocardial infarction',
                     'ISCIN': 'inferior myocardial ischemic',
                     'INJAS': 'subendocardial injury in anteroseptal leads',
                     'LMI': 'lateral myocardial infarction',
                     'ISCIL': 'inferolateral myocardial ischemic',
                     'LPFB': 'left posterior fascicular block',
                     'ISCAS': 'anteroseptal myocardial ischemic',
                     'INJAL': 'subendocardial injury in anterolateral leads',
                     'ISCLA': 'lateral myocardial ischemic',
                     'RVH': 'right ventricular hypertrophy',
                     'ANEUR': 'ST-T changes compatible with ventricular aneurysm',
                     'RAO/RAE': 'right atrial overload/enlargement',
                     'EL': 'electrolytic disturbance or drug (former EDIS)',
                     'WPW': 'Wolf-Parkinson-White syndrome',
                     'ILBBB': 'incomplete left bundle branch block',
                     'IPLMI': 'inferoposterolateral myocardial infarction',
                     'ISCAN': 'anterior myocardial ischemic',
                     'IPMI': 'inferoposterior myocardial infarction',
                     'SEHYP': 'septal hypertrophy',
                     'INJIN': 'subendocardial injury in inferior leads',
                     'INJLA': 'subendocardial injury in lateral leads',
                     'PMI': 'posterior myocardial infarction',
                     '3AVB': 'third degree atrioventricular block',
                     'INJIL': 'subendocardial injury in inferolateral leads',
                     '2AVB': 'second degree atrioventricular block',
                     'ABQRS': 'abnormal QRS(QRS changes)',
                     'PVC': 'ventricular premature complex',
                     'STD_': 'ST segment depression',
                     'VCLVH': 'voltage criteria (QRS) for left ventricular hypertrophy',
                     'QWAVE': 'Q waves present',
                     'LOWT': 'low amplitude T wave',
                     'NT_': 'T wave changes',
                     'PAC': 'atrial premature complex',
                     'LPR': 'prolonged PR interval',
                     'INVT': 'inverted T wave',
                     'LVOLT': 'low QRS voltages in the frontal and horizontal leads',
                     'HVOLT': 'high QRS voltage',
                     'TAB_': 'T wave abnormality',
                     'STE_': 'ST segment elevation',
                     'PRC(S)': 'premature complex(es)',
                     'SR': 'sinus rhythm',
                     'AFIB': 'atrial fibrillation',
                     'STACH': 'sinus tachycardia',
                     'SARRH': 'sinus arrhythmia',
                     'SBRAD': 'sinus bradycardia',
                     'PACE': 'normal functioning artificial pacemaker',
                     'SVARR': 'supraventricular arrhythmia',
                     'BIGU': 'bigeminal pattern (unknown origin, SV or Ventricular)',
                     'AFLT': 'atrial flutter',
                     'SVTAC': 'supraventricular tachycardia',
                     'PSVT': 'paroxysmal supraventricular tachycardia',
                     'TRIGU': 'trigeminal pattern (unknown origin, SV or Ventricular)'}
    generated_description_dict = {}
    for item in all_label_map.values():
        response = _generate_gemini_augment_(item)
        print(response)
        generated_description_dict[item] = response
    with open("ptbxl_label_map_description_gemini.json", "w") as f:
        json.dump(generated_description_dict, f)

def _generate_gemini_augment_(item):
    prompt_prefix_zeroshot = "I want you to play the role of a professional Electrocardiologist, and I need you to explain the meaning of "
    prompt_suffix_zeroshot = " in a 12-lead electrocardiogram report. Your answer must be less than 50 words."
    prompt_prefix_diagnosis = "I want you to play the role of a professional Electrocardiologist, and I need you to teach me how " \
                              "to diagnose "
    prompt_suffix_diagnosis = " from 12-lead ECG. such as what leads or what features to focus on ,etc. Your answer must be less " \
                                  "than 50 words."
    url = "CAHT_WITH_YOUR_GPT"
    headers = {"Content-Type": "application/json;charset=utf-8",
               "Accept": "*/*",
               "Accept-Encoding": "gzip, deflate, br",
               "Connection": "keep-alive"}
    # data = {"messages": prompt_prefix_diagnosis + item + prompt_suffix_diagnosis,
    #         "userId": "serveForPaper"}
    data = {"messages": prompt_prefix_zeroshot + item + prompt_suffix_zeroshot,
            "userId": "serveForPaper"}
    json_data = json.dumps(data)
    response = requests.post(url=url, data=json_data, headers=headers)
    return response.text


def generate_zhipuai_augment():
    """"""
    from zhipuai import ZhipuAI

    prompt_prefix_zeroshot = "I want you to play the role of a professional Electrocardiologist, and I need you to explain the meaning of "
    prompt_suffix_zeroshot = " in a 12-lead electrocardiogram report. Your answer must be less than 50 words."
    prompt_prefix_diagnosis = "I want you to play the role of a professional Electrocardiologist, and I need you to teach me how " \
                                  "to diagnose "
    prompt_suffix_diagnosis = " from 12-lead ECG. such as what leads or what features to focus on ,etc. Your answer must be less " \
                                  "than 50 words."

    client = ZhipuAI(api_key="YOUR_ZHIPU_API_KEY")

    all_label_map = {'NDT': 'non-diagnostic T wave abnormalities',
                     'NST_': 'ST segment changes',
                     'DIG': 'digitalis-effect',
                     'LNGQT': 'long QT interval',
                     'NORM': 'normal ECG',
                     'IMI': 'inferior myocardial infarction',
                     'ASMI': 'anteroseptal myocardial infarction',
                     'LVH': 'left ventricular hypertrophy',
                     'LAFB': 'left anterior fascicular block',
                     'ISC_': 'myocardial ischemic',
                     'IRBBB': 'incomplete right bundle branch block',
                     '1AVB': 'first degree atrioventricular block',
                     'IVCD': 'intraventricular conduction disturbance (block)',
                     'ISCAL': 'anterolateral myocardial ischemic',
                     'CRBBB': 'complete right bundle branch block',
                     'CLBBB': 'complete left bundle branch block',
                     'ILMI': 'inferolateral myocardial infarction',
                     'LAO/LAE': 'left atrial overload/enlargement',
                     'AMI': 'anterior myocardial infarction',
                     'ALMI': 'anterolateral myocardial infarction',
                     'ISCIN': 'inferior myocardial ischemic',
                     'INJAS': 'subendocardial injury in anteroseptal leads',
                     'LMI': 'lateral myocardial infarction',
                     'ISCIL': 'inferolateral myocardial ischemic',
                     'LPFB': 'left posterior fascicular block',
                     'ISCAS': 'anteroseptal myocardial ischemic',
                     'INJAL': 'subendocardial injury in anterolateral leads',
                     'ISCLA': 'lateral myocardial ischemic',
                     'RVH': 'right ventricular hypertrophy',
                     'ANEUR': 'ST-T changes compatible with ventricular aneurysm',
                     'RAO/RAE': 'right atrial overload/enlargement',
                     'EL': 'electrolytic disturbance or drug (former EDIS)',
                     'WPW': 'Wolf-Parkinson-White syndrome',
                     'ILBBB': 'incomplete left bundle branch block',
                     'IPLMI': 'inferoposterolateral myocardial infarction',
                     'ISCAN': 'anterior myocardial ischemic',
                     'IPMI': 'inferoposterior myocardial infarction',
                     'SEHYP': 'septal hypertrophy',
                     'INJIN': 'subendocardial injury in inferior leads',
                     'INJLA': 'subendocardial injury in lateral leads',
                     'PMI': 'posterior myocardial infarction',
                     '3AVB': 'third degree atrioventricular block',
                     'INJIL': 'subendocardial injury in inferolateral leads',
                     '2AVB': 'second degree atrioventricular block',
                     'ABQRS': 'abnormal QRS(QRS changes)',
                     'PVC': 'ventricular premature complex',
                     'STD_': 'ST segment depression',
                     'VCLVH': 'voltage criteria (QRS) for left ventricular hypertrophy',
                     'QWAVE': 'Q waves present',
                     'LOWT': 'low amplitude T wave',
                     'NT_': 'T wave changes',
                     'PAC': 'atrial premature complex',
                     'LPR': 'prolonged PR interval',
                     'INVT': 'inverted T wave',
                     'LVOLT': 'low QRS voltages in the frontal and horizontal leads',
                     'HVOLT': 'high QRS voltage',
                     'TAB_': 'T wave abnormality',
                     'STE_': 'ST segment elevation',
                     'PRC(S)': 'premature complex(es)',
                     'SR': 'sinus rhythm',
                     'AFIB': 'atrial fibrillation',
                     'STACH': 'sinus tachycardia',
                     'SARRH': 'sinus arrhythmia',
                     'SBRAD': 'sinus bradycardia',
                     'PACE': 'normal functioning artificial pacemaker',
                     'SVARR': 'supraventricular arrhythmia',
                     'BIGU': 'bigeminal pattern (unknown origin, SV or Ventricular)',
                     'AFLT': 'atrial flutter',
                     'SVTAC': 'supraventricular tachycardia',
                     'PSVT': 'paroxysmal supraventricular tachycardia',
                     'TRIGU': 'trigeminal pattern (unknown origin, SV or Ventricular)'}
    generated_description_dict = {}

    for item in all_label_map.values():
        if item in generated_description_dict.keys():
            continue
        response = client.chat.completions.create(
            model="glm-4",  # 填写需要调用的模型名称
            messages=[
                {"role": "user", "content": prompt_prefix_diagnosis + item + prompt_suffix_diagnosis},
                # {"role": "user", "content": prompt_prefix_zeroshot + item + prompt_suffix_zeroshot},
            ],
        )
        response = response.choices[0].message.content
        print(response)
        generated_description_dict[item] = response

    with open("ptbxl_label_map_report_zhipuai.json", "w") as f:
        json.dump(generated_description_dict, f)



if __name__ == '__main__':
    # PTB - xl raw data storage paths:
    datafolder = '/data_C/sdb1/lyi/ECGFM-KED-main/dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
    experiments = [
        #('exp0', 'all'),  # (21799, 71)
        #('exp1', 'diagnostic'), # (21388, 44)
        #('exp1.1', 'subdiagnostic'),     # (21388, 23)
        ('exp1.1.1', 'superdiagnostic'),
        #('exp2', 'form'),   # (8978, 19)
        #('exp3', 'rhythm')  # (21030, 12)
    ]
    for name, task in experiments:
        handler_data(name, task, datafolder)
