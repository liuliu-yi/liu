import pandas as pd

def get_wave_info(data):
    keys = ['RR_Interval', 'PR_Interval', 'QRS_Complex', 'QT_Interval',
            'QTc_Interval', 'P_Wave_Peak', 'R_Wave_Peak', 'T_Wave_Peak']
    text_describe = ""
    text_describe += f" RR: {data['RR_Interval']}"
    text_describe += f" PR: {data['PR_Interval']}"
    text_describe += f" QRS: {data['QRS_Complex']}"
    text_describe += f" QT/QTc: {data['QT_Interval']}/{data['QTc_Interval']}"
    text_describe += f" P/R/T Wave: {data['P_Wave_Peak']}/{data['R_Wave_Peak']}/{data['T_Wave_Peak']}"
    return text_describe

def find_wave_features(subject_id, study_id, mm_df):
    """
    根据subject_id和study_id查找machine_measurements.csv中对应的那一行的波形特征。
    """
    row = mm_df[(mm_df['subject_id'] == str(subject_id)) & (mm_df['study_id'] == str(study_id))]
    if row.empty:
        return None
    row = row.iloc[0]
    # 你需要根据你的csv实际字段名调整键名
    data = {
        'RR_Interval': row.get('RR_Interval', 'NA'),
        'PR_Interval': row.get('PR_Interval', 'NA'),
        'QRS_Complex': row.get('QRS_Complex', 'NA'),
        'QT_Interval': row.get('QT_Interval', 'NA'),
        'QTc_Interval': row.get('QTc_Interval', 'NA'),
        'P_Wave_Peak': row.get('P_Wave_Peak', 'NA'),
        'R_Wave_Peak': row.get('R_Wave_Peak', 'NA'),
        'T_Wave_Peak': row.get('T_Wave_Peak', 'NA'),
    }
    return data




# 示例用法:
# 用法示例
# 1. 读取new_record_list.csv为DataFrame
mm_df = pd.read_csv('/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/mimiciv/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/new_record_list.csv', dtype=str)

#2. 读取ked_data_y_total_test.json
import json
with open('/data_C/sdb1/lyi/ked/ECGFM-KED-main/dataset/mimiciv/processed1/data_y_total_test.json', 'r') as f:
    report_data = json.load(f)


# 3. 对每条报告数据，拼接波形特征描述
for item in report_data:
    subject_id = item['subject_id']
    study_id = item['study_id']
    wave_data = find_wave_features(subject_id, study_id, mm_df)
    if wave_data is not None:
        wave_desc = get_wave_info(wave_data)
        item['report'] = item['report'].rstrip('.') + ". Wave features:" + wave_desc

# 4. 保存新数据
with open('ked_data_y_total_test_with_wave.json', 'w', encoding='utf8') as f:
    json.dump(report_data, f, ensure_ascii=False, indent=2)




