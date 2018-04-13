import anonymizer
import os
import csv

mappings = {
        "*\\{}\\*\\RAPID_37\\*\\*\\RAPID_MTT": "{}\\Ct2_Cerebrale\\",
        "*\\{}\\*\\RAPID_37\\*\\*\\RAPID_rCBF": "{}\\Ct2_Cerebrale\\",
        "*\\{}\\*\\RAPID_37\\*\\*\\RAPID_rCBV": "{}\\Ct2_Cerebrale\\",
        "*\\{}\\*\\RAPID_37\\*\\*\\RAPID_Tmax": "{}\\Ct2_Cerebrale\\",
        "*\\{}\\*\\SPC_301mm_Std": "{}\\Ct2_Cerebrale\\",
        "*\\{}\\Neuro*\\t2_tse*": "{}\\Neuro_Cerebrale_64Ch\\",
        "*\\{}\\VOI*" : "{}\\"
}

main_dir = "d:"
patients = []
for folder in os.listdir(main_dir):
    if anonymizer.is_patient(os.path.join(main_dir, folder)):
        patients.append(folder)


patient_mapping = []
for patient in patients:
    mapping_patient = mappings.copy()
    mapping_patient_conv = {}
    for k, v in mapping_patient.items():
        k = k.replace("{}", patient)
        v = v.replace("{}", patient)
        mapping_patient_conv.update({k:v})
    patient_mapping.append(mapping_patient_conv)


filename = "patient_mappings.csv"
for dict_patient in patient_mapping:
    with open(filename, 'a') as f:
        w = csv.writer(f)
        w.writerows(dict_patient.items())

