import os
import argparse
import pandas as pd
from unidecode import unidecode
import datetime
import csv

logfile = "log-list-files-{}.txt".format(datetime.datetime.now())

def date2string(date):
    if date is None or str(date) is "NaT":
        return None
    y = str(date.year)
    m = str(date.month)
    d = str(date.day)
    if len(m)<2:
        m = "0"+m
    if len(d)<2:
        d = "0"+d
    return y+m+d


def get_RAPID_folder(patient_folder, date_CT):
    return get_conditioned_file(patient_folder, date_CT, "rapid_37")

def get_SPC_folder(patient_folder, date_CT):
    conditions = ["SPC_301mm_Std_complet","SPC_301mm_Std", "DE_SPC_30", "DE_SPC_VNC_30"]
    paths = []
    for c in conditions:
        path = get_conditioned_file(patient_folder, date_CT, c)
        paths.append(path)


    paths = [x for x in paths if x is not None]
    if len(paths)==0:
        return None

    # Prioritize
    for p in paths:
        if "SPC_301mm_Std_complet".lower() in p.lower():
            return p
        elif "SPC_301mm_Std".lower() in p.lower():
            return p

    # return default
    return paths[0]



def get_T2_folder(patient_folder, date_MRI):
    conditions = ["t2_tse", "t2w_tse_tra", "t2w_tse_ax"]
    returned_folder = None
    for c in conditions:
        path = get_conditioned_file(patient_folder, date_MRI, c)
        if path is not None:
            if returned_folder is None:
                returned_folder = path
            else:
                raise Exception("Duplicate : {} vs {}".format(path, returned_folder))

    return returned_folder


def get_ADC_folder(patient_folder, date_MRI):
    conditions = ["adc", "dDWI_HR"]
    returned_folder = None
    for c in conditions:
        path = get_conditioned_file(patient_folder, date_MRI, c)
        if path is not None:
            if returned_folder is None:
                returned_folder = path
            else:
                raise Exception("Duplicate : {} vs {}".format(path, returned_folder))

    return returned_folder


def get_TRACEW_folder(patient_folder, date_MRI):
    conditions = ["tracew", "eDWI_tra", "DWI_HR", "isoDWI"]
    returned_folder = None
    for c in conditions:
        path = get_conditioned_file(patient_folder, date_MRI, c)
        if path is not None:
            if returned_folder is None:
                returned_folder = path
            else:
                raise Exception("Duplicate : {} vs {}".format(path, returned_folder))

    return returned_folder


def get_lesion(patient_folder):
    files = os.listdir(patient_folder)
    lesion = None
    for f in files:
        if f.startswith("VOI"):
            lesion = os.path.join(patient_folder, f)
    return lesion


def get_conditioned_file(patient_folder, date, condition):
    returned_folder = None

    folders = os.listdir(patient_folder)
    for f in folders:
        if date in f:
            tmp_folder = os.path.join(patient_folder, f)
            if os.path.isdir(tmp_folder):
                fs = os.listdir(tmp_folder)
                for subf in fs:
                    if condition.lower() in subf.lower():
                        if returned_folder is None:
                            returned_folder = os.path.join(patient_folder, f, subf)
                        else:
                            print("DUPLICATED {} : {} vs {}".format(condition, os.path.join(patient_folder, f, subf),
                                                                    returned_folder))
                            return None
    return returned_folder


def is_valid(patient_dict, modalities=["RAPID","SPC","T2","ADC","TRACEW", "LESION"]):
    valid = True
    for k,v in patient_dict.items():
        if k in modalities:
            if patient_dict[k] is None:
                valid = False

    return valid


def create_transfer_file(valid_patient_dicts, main_dir, filename="patient_mappings.csv",
                         modalities=["RAPID","SPC","T2","ADC","TRACEW", "LESION"]):
    subfolders_to_keep = {
        "RAPID": [os.path.join("{}",  "*", "*", "RAPID_MTT"),
                  os.path.join("{}", "*", "*", "RAPID_rCBF"),
                  os.path.join("{}", "*", "*", "RAPID_rCBV"),
                  os.path.join("{}", "*", "*", "RAPID_Tmax")],
        "SPC": [os.path.join("{}")],
        "T2": [os.path.join("{}")],
        "ADC": [os.path.join("{}")],
        "TRACEW": [os.path.join("{}")],
        "LESION": [os.path.join("{}")]
    }

    mappings = {
        "RAPID":  os.path.join("{}", "Ct2_Cerebrale"),
        "SPC":  os.path.join("{}", "Ct2_Cerebrale"),
        "T2":  os.path.join("{}", "Neuro_Cerebrale_64Ch"),
        "TRACEW": os.path.join("{}", "Neuro_Cerebrale_64Ch"),
        "ADC" : os.path.join("{}", "Neuro_Cerebrale_64Ch"),
        "LESION": os.path.join("{}")
    }

    mapping_patient_conv = {}

    for d in valid_patient_dicts:
        patient_folder = d["PATIENT_FOLDER"]
        for k,v in d.items():
            if k in subfolders_to_keep and k in mappings and k in modalities:
                subfolders = subfolders_to_keep[k]
                for subf in subfolders:
                    input_folder = subf.format(d[k]).replace(main_dir,"*")
                    output_folder = mappings[k].format(patient_folder).replace(main_dir,"")
                    if output_folder.startswith("/"):
                        output_folder = output_folder[1:]
                    mapping_patient_conv[input_folder]= output_folder
                    #print("{} -> {}".format(input_folder, output_folder))


    with open(filename, 'w') as f:
        w = csv.writer(f)
        w.writerows(mapping_patient_conv.items())




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter and copy files.')
    parser.add_argument("-p", "--path", type=str, help="Main directory path", default="/media/exfat/stroke_db/2016/2016_without_VOI")
    parser.add_argument("-f", "--history_patient_file", type=str, help="File", default="/home/snarduzz/Downloads/orig_cleaned_clinical_data_2016.xlsx")
    parser.add_argument("-m", "--modalities", help="Modalities to fulfill", default=["RAPID", "SPC", "T2","ADC","TRACEW", "LESION"])

    args = parser.parse_args()
    history_file = args.history_patient_file
    path = args.path
    modalities = args.modalities
    # TODO REMOVE
    modalities = ["T2", "RAPID", "SPC"]
    filename = "patient_mappings-strokedb-without.csv"

    df = pd.read_excel(history_file)
    patients = [x for x in os.listdir(args.path) if not x.startswith(".")]

    PATIENTS_OK = []
    PATIENTS_LESION = []
    PATIENT_VALID = []

    for id, row in df.iterrows():

        surname = row['Nom'].strip().replace("-","").replace(" ","_")
        name = row['Prénom'].strip().replace("-","").replace(" ","_")

        first_type = row['firstimage_type']
        follow_type = row['followimage type 24h']
        other_type = row['other followup type']

        first_date = row['Firstimage_date']
        follow_date = row['Followimage_date 24h']
        other_date = row['other followup type_date']

        first_date = date2string(first_date)
        follow_date = date2string(follow_date)
        other_date = date2string(other_date)

        if other_type is None:
            print("OTHER IS NONE : ", surname, name)


        if (first_type == "CT" and follow_type == "MRI") \
                or (first_type == "CT" and other_type =="MRI")\
                or (follow_type == "CT" and other_type == "CT"):
            print(surname, name)
            # find main folder
            patient_folder = None
            for p in patients:
                p = unidecode(p)
                if (unidecode(surname).lower() in p.lower() and unidecode(name).lower() in p.lower()):
                    patient_folder = os.path.join(path,p)

            # find CT  and MRI folder
            date_CT = None
            date_MRI = None

            # standard case

            if first_type == "CT" and follow_type == "CT" and other_type == "MRI":
                # get most recent CT
                date_CT = max(first_date, follow_date)
                date_MRI = other_date

            elif first_type == "CT" and follow_type =="MRI":
                date_CT = first_date
                date_MRI = follow_date
            else:
                print("SPECIAL CASE: {} {}".format(surname, name))
                valid_dates = [first_date, follow_date, other_date]
                valid_dates = [x for x in valid_dates if x is not None]
                if len(valid_dates)<2:
                    continue

                date_CT = min(valid_dates)
                date_MRI = max(valid_dates)
                first_date = date_CT

            if date_CT is None:
                print("ERROR : ",surname, name, "\n")
                continue

            RAPID = get_RAPID_folder(patient_folder, date_CT)
            SPC = get_SPC_folder(patient_folder, date_CT)

            # if not recent RAPID
            if RAPID is None:
                date_CT = first_date
                RAPID = get_RAPID_folder(patient_folder, first_date)
            # If really not found SPC, try with the first date
            if SPC is None:
                SPC = get_SPC_folder(patient_folder, first_date)

            T2 = get_T2_folder(patient_folder, date_MRI)
            ADC = get_ADC_folder(patient_folder, date_MRI)
            TRACEW = get_TRACEW_folder(patient_folder,date_MRI)
            LESION = get_lesion(patient_folder)

            dict = {
                "NAME": name,
                "SURNAME":surname,
                "RAPID" : RAPID,
                "SPC": SPC,
                "T2": T2,
                "TRACEW": TRACEW,
                "ADC":ADC,
                "LESION":LESION,
                "DATE_CT":date_CT,
                "DATE_MRI":date_MRI,
                "PATIENT_FOLDER":patient_folder,
            }

            if is_valid(dict, modalities=modalities):
                PATIENT_VALID.append(dict)

            OK = ""
            messages = []

            if is_valid(dict, modalities):
                if LESION:
                    OK = "OK"
                    PATIENTS_LESION.append("{}_{}".format(name,surname))
                else:
                    OK = "OK - MISSING LESION"
                    PATIENTS_OK.append("{}_{}".format(name,surname))
            if LESION is None:
                    OK = "MISSING LESION"

            messages.append("---- {} {} ---- {}".format(surname, name, OK))
            messages.append(
                "TIMELINE : {} ({}) -> {} ({}) -> {} ({})".format(first_type, first_date, follow_type, follow_date, other_type,
                                                       other_date))
            messages.append("CT : {} ({})".format(RAPID, date_CT))
            messages.append("SPC : {} ({})".format(SPC, date_CT))
            messages.append("T2 : {} ({})".format(T2, date_MRI))
            messages.append("TRACEW : {} ({})".format(TRACEW, date_MRI))
            messages.append("ADC : {} ({})".format(ADC, date_MRI))
            messages.append("Lesion : {}".format(LESION))
            messages.append("Patient folder : {}".format(patient_folder))
            messages.append("")
            messages.append("")

            with open(logfile, "a") as f:
               lines = '\n'.join(messages)
               f.writelines(lines)

    print("")
    print("PATIENT LESION :")
    for p in PATIENTS_LESION:
        print(p)
    print("TOTAL OK WITH LESION : {}".format(len(PATIENTS_LESION)))
    print("")
    print("PATIENT OK :")
    for p in PATIENTS_OK:
        print(p)
    print("TOTAL OK WITHOUT LESION : {}".format(len(PATIENTS_OK)))


    print("VALID {}".format(len(PATIENT_VALID)))

    create_transfer_file(PATIENT_VALID, main_dir=path, filename=filename, modalities=modalities)