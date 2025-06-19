# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-06-20 15:39

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import pandas as pd
import numpy as np
from scipy.signal import resample
import wfdb
import ast
def handler_data():
    """
        before running this data preparing code,
        please first download the raw data from https://doi.org/10.6084/m9.figshare.c.4560497.v2,
        and put it in data_path
        """
    """"""


    def str_to_list(x):
        if isinstance(x, list):
            return x
        try:
            return ast.literal_eval(x)
        except:
            return [x]
    #信号归一化
    def zscore_norm(ecg_signal):
            # (channels, timesteps)
            mean = np.mean(ecg_signal, axis=1, keepdims=True)
            std = np.std(ecg_signal, axis=1, keepdims=True)
            ecg_norm=(ecg_signal - mean) / (std + 1e-8)
            return ecg_norm

    row_data_file = pd.read_csv('/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/shaoxing/new_diagnostics.csv')
    print(row_data_file.shape)
    print(row_data_file.head())

    # 读取SNOMED映射表，建立映射字典
    map_df = pd.read_csv('/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/shaoxing/CSD/ConditionNames_SNOMED-CT.csv')
    snomed_map = dict(zip(map_df['Snomed_CT'].astype(str), map_df['Full Name']))
   
    # 先将所有的signal读取出来
    signal_data = []
    error_files = []
    error_index = []

    for idx, item in row_data_file.iterrows():
        # 注意：filename 字段通常不带扩展名
        base_path = '/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/shaoxing/CSD/'
        file_path = base_path + item['filename']
        # Example: WFDBRecords/01/010/JS00001
        try:
            record = wfdb.rdrecord(file_path)
            sig = record.p_signal  # shape: (length, leads)
            #信号自归一化
            #sig = zscore_norm(sig)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            error_files.append(file_path)
            error_index.append(idx)
            sig = np.zeros((5000, 12))  # 占位
        signal_data.append(sig)

    # 剔除无效信号
    signal_data = np.array(signal_data)
    if error_index:
        signal_data = np.delete(signal_data, error_index, axis=0)
        row_data_file = row_data_file.drop(error_index).reset_index(drop=True)

    # 用Snomed_CT做多标签处理
    # 多标签分割
    row_data_file['Snomed_CT_list'] = row_data_file['Snomed_CT'].apply(
        lambda x: [s.strip() for s in str(x).split(',') if s.strip()]
    )
    # 可根据实际标签分隔符调整

    # 统计标签分布，过滤少见标签
    counts = pd.Series([l for sub in row_data_file['Snomed_CT_list'] for l in sub]).value_counts()
    valid_labels = counts[counts > 100].index  # 只保留样本数大于100的标签
    row_data_file['Snomed_CT_list'] = row_data_file['Snomed_CT_list'].apply(
        lambda x: list(set(x).intersection(set(valid_labels)))
    )
    row_data_file['Label_len'] = row_data_file['Snomed_CT_list'].apply(lambda x: len(x))
    
    # 只保留有有效标签的样本
    mask = row_data_file['Label_len'] > 0
    sig = signal_data[mask]
    data_file = row_data_file[mask]
    

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

    # 新建 report_wave 列，拼接报告和波形特征
    def append_wave_to_report(row):
        return str(row['report']) + "." + get_wave_info(row)

    data_file['report_wave'] = data_file.apply(append_wave_to_report, axis=1)
    
    data_file['Snomed_CT_list'] = data_file['Snomed_CT_list'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
    )
  
    # Snomed_CT批量映射为多标签全称
    def snomed_to_names(code_list):
        return [snomed_map.get(str(code), str(code)) for code in code_list]
    
    data_file['label']= data_file['Snomed_CT_list'].apply(snomed_to_names)

    #在划分之前保存一下 已经预处理好的完整表 方便后续统计标签
    data_file.to_csv(f'/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/shaoxing/data/processed_new_diagnostics.csv')

    data_file=pd.read_csv('/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/shaoxing/data/processed_new_diagnostics.csv')
    # 按8:1:1随机划分训练、验证、测试集
    # 保证sig, label, report顺序一致
    X_temp, y_test = train_test_split(
        data_file, test_size=0.1, random_state=42, stratify=None
    )
    val_ratio = 0.1 / 0.9
    y_train, y_val = train_test_split(
        X_temp, test_size=val_ratio, random_state=42, stratify=None
    )

    # 保存
    #信号路径
    root_dir = '/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/shaoxing/CSD/'
    #拼接为完整路径
    path_train = root_dir + y_train['filename'].astype(str)
    path_val = root_dir + y_val['filename'].astype(str)
    path_test = root_dir + y_test['filename'].astype(str)

    path_train.to_csv(f'/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/shaoxing/data/ecg_train.csv',index=False)
    path_val.to_csv(f'/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/shaoxing/data/ecg_val.csv',index=False)
    path_test.to_csv(f'/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/shaoxing/data/ecg_test.csv',index=False)

    #标签
    # 将每一行的list转为字符串再保存
    y_train.astype(str).to_csv(f'/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/shaoxing/data/label_train.csv',index=False) #默认保留索引与表头
    y_val.astype(str).to_csv(f'/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/shaoxing/data/label_val.csv',index=False)
    y_test.astype(str).to_csv(f'/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/shaoxing/data/label_test.csv',index=False)
    #报告
    report_train.to_csv(f'/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/shaoxing/data/report_train.csv',index=False)
    report_val.to_csv(f'/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/shaoxing/data/report_val.csv',index=False)
    report_test.to_csv(f'/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/shaoxing/data/report_test.csv',index=False)


   

if __name__ == '__main__':
    handler_data()
    # f = open('/home/tyy/unECG/dataset/shaoxing/mlb.pkl', 'rb')
    # data = pickle.load(f)
    # print(data.classes_)

    # item_count = []
    # data = np.load('signal_data.npy', allow_pickle=True)
    # for item in data:
    #     if item.sum() == 0:
    #         item_count.append(item)
    # print(len(item_count))
    # print(data.shape)