# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-06-20 15:39

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import pandas as pd
import numpy as np
from signal_analysis import signal_analyse,calculate_waveforms

def handler_data():
    """
        before running this data preparing code,
        please first download the raw data from https://doi.org/10.6084/m9.figshare.c.4560497.v2,
        and put it in data_path
        """
    """"""


    row_data_file = pd.read_csv('/data_C/sdb1/lyi/ECG-Chat-master/data/champan-shaoxing/diagnostics.csv')
    print(row_data_file.shape)
    print(row_data_file.head())

    # 先将所有的signal读取出来
    signal_data = []
    error_files = []
    error_index = []

    for idx, item in row_data_file.iterrows():
        # 注意：filename 字段通常不带扩展名
        base_path = '/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/shaoxing'
        file_path = base_path + item['filename']
        # Example: WFDBRecords/01/010/JS00001
        try:
            record = wfdb.rdrecord(file_path)
            sig = record.p_signal  # shape: (length, leads)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            error_files.append(file_path)
            error_index.append(idx)
            sig = np.zeros((5000, 12))  # 占位
        signal_data.append(sig)

    #计算波形信息
    data_dict = calculate_waveforms(signal_data, sampling_rate=500, lead_index=1)
    for key in data_dict.keys():
        row_data_file[key] = data_dict[key]
        row_data_file.to_csv("/data_C/sdb1/lyi/ECG-Chat-master/data/champan-shaoxing/new_diagnostics.csv") #得到含波形信息的原表

   

if __name__ == '__main__':
    handler_data()
   