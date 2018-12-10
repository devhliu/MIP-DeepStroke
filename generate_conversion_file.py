import anonymizer
import os
import csv

mappings = {
        os.path.join("*", "{}", "*", "RAPID_37", "*", "*", "RAPID_MTT"): os.path.join("{}", "Ct2_Cerebrale"),
         os.path.join("*", "{}", "*", "RAPID_37", "*", "*", "RAPID_rCBF") : os.path.join("{}", "Ct2_Cerebrale"),
         os.path.join("*", "{}", "*", "RAPID_37", "*", "*", "RAPID_rCBV") : os.path.join("{}", "Ct2_Cerebrale"),
         os.path.join("*", "{}", "*", "RAPID_37", "*", "*", "RAPID_Tmax") : os.path.join("{}", "Ct2_Cerebrale"),
         os.path.join("*", "{}", "*", "SPC_301*Std*") : os.path.join("{}", "Ct2_Cerebrale"),
         os.path.join("*", "{}", "*", "DE_SPC_30*Std*") : os.path.join("{}", "Ct2_Cerebrale"),
         os.path.join("*", "{}", "*Neuro*", "t2*tse*tra") : os.path.join("{}", "Neuro_Cerebrale_64Ch"),
         os.path.join("*", "{}", "VOI*") : os.path.join("{}"),
         os.path.join("*", "{}", "*Neuro*", "*TRACEW*") : os.path.join("{}", "Neuro_Cerebrale_64Ch"),
         os.path.join("*", "{}", "*Neuro*", "*ADC*") : os.path.join("{}", "Neuro_Cerebrale_64Ch"),
}

main_dir = "/media/exfat"
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

