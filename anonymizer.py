import os
import subprocess
import re
import os
import numpy as np
import progress
from progress.bar import IncrementalBar
import shutil
from distutils.dir_util import copy_tree
import hashlib
import csv

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
    return number, patient_data


def filename_contains_patient_data(filename, patient_data):
    for info in patient_data:
        if info in filename:
            return True
    return False


def anonymize_string(string, patient_data, patient_hash):
    string_name = string[:] #Copy
    for x in patient_data:
        # replace all data except patient number
        if patient_hash not in x:
            string_name = string_name.replace(x,"_")


    # leave at least patient number
    if patient_hash not in string_name:
        string_name = str(patient_hash)+"_"+string_name
    anonymized_string = string_name.strip("_")

    return anonymized_string


def anonymize_folder(folder):
    base_name = os.path.basename(folder)
    _, patient_data = get_patient_number(base_name)
    patient_hash = hashlib.sha512(base_name.encode())
    patient_hash = patient_hash.hexdigest()

    for dir, subdirs, files in os.walk(folder):
        # Anonymize subfile
        for file in files:
            if filename_contains_patient_data(file, patient_data):
                file_anonymized = anonymize_string(file, patient_data, patient_hash)
                oldfile = os.path.join(dir, file)
                newfile = os.path.join(dir, file_anonymized)
                os.rename(oldfile, newfile)

        # Anonymize subfolders
        for subdir in subdirs:
            # If the patient number and name appears in a folder, rename id
            if filename_contains_patient_data(subdir, patient_data):
               subdir_anonymized = anonymize_string(subdir, patient_data, patient_hash)
               olddir = os.path.join(dir, subdir)
               newdir = os.path.join(dir, subdir_anonymized)
               os.rename(olddir, newdir)

    new_root_name = anonymize_string(base_name, patient_data, patient_hash)
    new_root_dir = folder.replace(base_name, new_root_name)
    os.rename(folder, new_root_dir)

    return patient_hash

def copy_to(main_dir, output_dir):
    # Get Subjects folders
    folders = os.listdir(main_dir)
    subjects = [folder_name for folder_name in folders if(is_patient(folder_name))]
    not_subjects = [folder_name for folder_name in folders if not (is_patient(folder_name))]

    bar = IncrementalBar(max=len(subjects))
    total_size_need = 0
    print("Estimating required size:")
    for subject in subjects[:3]:
        size = get_size(os.path.join(main_dir, subject))
        total_size_need += size
        bar.next()
    bar.finish()
    print(total_size_need)

    free_space = get_free_space(output_dir)
    if(free_space<total_size_need):
        raise Exception("No enough space available in {}".format(output_dir))

    print("Copy files in output folder:")
    bar_copy = IncrementalBar(max=len(subjects))
    for subject in subjects[:3]:
        subject_path = os.path.join(main_dir, subject)
        output_subject_path = os.path.join(output_dir, subject)
        copy_tree(subject_path, output_subject_path)
        bar_copy.next()
    bar_copy.finish()

    return output_dir


def get_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

if __name__ == '__main__':
    IN_PLACE = True
    main_dir = "d:\\Anonymized_Data"
    # data_dir = os.path.join(main_dir, 'working_data')
    output_dir = os.path.join(main_dir, 'Anonymized_Data')
    dcm2niix_path = "c:\\Program Files\\dcm2niix\\dcm2niix.exe"
    patient_data_file = "c:"

    if not IN_PLACE:
        main_dir = copy_to(main_dir, output_dir)

    dict_hash = dict()
    for folder in os.listdir(main_dir):
        try:
            patient_hash = anonymize_folder(os.path.join(main_dir, folder))
            dict_hash.update({folder:patient_hash})
        except Exception as e:
            print(e)
            print("Error anonymizing the following folder:")

    filename = "hash_patients.csv"
    with open(filename, 'a') as f:
        w = csv.writer(f)
        w.writerows(dict_hash.items())