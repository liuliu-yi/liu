import pandas as pd
import numpy as np
import wfdb
import ast

def load_raw_data(df, sampling_rate, path):# 加载 ECG 信号数据。
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data
#PTB-XL 数据集的路径。
path = '/data_C/sdb1/lyi/ECGFM-KED-main/dataset/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
sampling_rate=100
 #加载并转换注释数据
# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data 加载原始信号数据
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation加载诊断聚合信息,筛选出诊断类别的语句。
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):#将原始诊断代码聚合为诊断大类。
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass将诊断大类添加到 DataFrame Y 中。
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Split data into train and test划分训练集和测试集
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)] #训练集的 ECG 信号数据。
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass #训练集的诊断大类标签
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
