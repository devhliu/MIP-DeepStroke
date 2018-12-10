import os
from tqdm import tqdm
import shutil
from distutils.dir_util import copy_tree
import csv
import pandas as pd

database_patients = "/home/snarduzz/patient_db.xlsx"

def is_patient(folder):
    try:
        number = folder.split("_")[::-1][0]
        if(len(number)!=8 or not number.isdigit()):
            raise Exception()
    except:
        return False
    return True


def get_free_space(output_dir):
    total, used, free = shutil.disk_usage(output_dir)
    return free


def get_used_space(directory):
    total, used, free = shutil.disk_usage(directory)
    return used


def get_patient_number(folder):
    # Assume format is "NAME_FORNAME_NUMBER"
    number = folder.split("_")[::-1][0]
    patient_data = folder.split("_")

    day = number[-2:]
    month = number[-4:-2]
    year = number[-8:-4]
    birthdate = "{}.{}.{}".format(day, month,year)
    patient_data.append(birthdate)
    df = pd.read_csv(database_patients, sep=";")
    df['name'] = df['name'].str.lower()
    df['first_name'] = df['first_name'].str.lower()
    df['birth_date'] = df['birth_date'].str.lower()

    df_selected = pd.DataFrame(columns=df.columns)

    for data in patient_data[:2]:
        data = str(data).lower()
        selected = df[(df.name.str.contains(data))
                | (df.first_name.str.contains(data))
                | (df.birth_date.str.contains(data))]

        df_selected = pd.concat([df_selected,selected])
        if len(df_selected) == 1:
            break
    df_selected = df_selected[['name','first_name','birth_date','id_hospital_case']]
    df_selected = df_selected.drop_duplicates()
    df_selected['score']=0
    #score
    for id,row in df_selected.iterrows():
        score = 0
        for attribute in row:
            for data in patient_data:
                if data in str(attribute):
                    score += 1
        df_selected.loc[id,'score'] = score

    df_selected = df_selected.sort_values(by='score',ascending=False)
    if len(df_selected)<1:
        print("could not retrieve patient number for patient : {}".format(patient_data))
    if len(df_selected)>1:
        print("Multiple results for patient {}".format(patient_data))
        #print(df_selected)
    number = str(df_selected['id_hospital_case'].values[0])

    return number, patient_data


def filename_contains_patient_data(filename, patient_data):
    for info in patient_data:
        if info in filename:
            return True
    return False


def anonymize_string(string, patient_data, patient_number):
    string_name = string[:] #Copy


    for x in patient_data:
        # replace all data except patient number
        if patient_number not in x:
            string_name = string_name.replace(x,"_")


    # leave at least patient number
    if patient_number not in string_name:
        string_name = str(patient_number)+"_"+string_name
    anonymized_string = string_name.strip("_")

    return anonymized_string


def anonymize_folder(folder):
    base_name = os.path.basename(folder)
    patient_number, patient_data = get_patient_number(base_name)

    for dir, subdirs, files in os.walk(folder):
        # Anonymize subfile
        for file in files:
            if filename_contains_patient_data(file, patient_data):
                file_anonymized = anonymize_string(file, patient_data, patient_number)
                oldfile = os.path.join(dir, file)
                newfile = os.path.join(dir, file_anonymized)
                print(oldfile, "->", newfile)
                os.rename(oldfile, newfile)

        # Anonymize subfolders
        for subdir in subdirs:
            # If the patient number and name appears in a folder, rename id
            if filename_contains_patient_data(subdir, patient_data):
               subdir_anonymized = anonymize_string(subdir, patient_data, patient_number)
               olddir = os.path.join(dir, subdir)
               newdir = os.path.join(dir, subdir_anonymized)
               print(olddir, "->", newdir)
               os.rename(olddir, newdir)

    new_root_name = anonymize_string(base_name, patient_data, patient_number)
    new_root_dir = folder.replace(base_name, new_root_name)
    print(folder,"->",new_root_dir)
    os.rename(folder, new_root_dir)

    return patient_number

def copy_to(main_dir, output_dir):
    # Get Subjects folders
    folders = os.listdir(main_dir)
    subjects = [folder_name for folder_name in folders if(is_patient(folder_name))]
    not_subjects = [folder_name for folder_name in folders if not (is_patient(folder_name))]

    total_size_need = 0
    print("Estimating required size:")
    for subject in tqdm(subjects[:3]):
        size = get_size(os.path.join(main_dir, subject))
        total_size_need += size

    print(total_size_need)

    free_space = get_free_space(output_dir)
    if(free_space<total_size_need):
        raise Exception("No enough space available in {}".format(output_dir))

    print("Copy files in output folder:")
    for subject in tqdm(subjects[:3]):
        subject_path = os.path.join(main_dir, subject)
        output_subject_path = os.path.join(output_dir, subject)
        copy_tree(subject_path, output_subject_path)

    return output_dir


def get_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

if __name__ == '__main__':
    IN_PLACE = False
    main_dir = "/media/exfat"
    # data_dir = os.path.join(main_dir, 'working_data')
    output_dir = os.path.join(main_dir, 'Anonymized_Data')

    if not IN_PLACE:
        main_dir = copy_to(main_dir, output_dir)

    dict_hash = dict()
    for folder in os.listdir(main_dir):
        try:
            if("_" not in folder):
                print(folder, "already anonymized")
            else:
                patient_number = anonymize_folder(os.path.join(main_dir, folder))
                dict_hash.update({folder:patient_number})
        except Exception as e:
            print(e.with_traceback())
            print("Error anonymizing the following folder:")

    filename = os.path.join(main_dir, "anonymized_patients.csv")
    with open(filename, 'a') as f:
        w = csv.writer(f)
        w.writerows(dict_hash.items())