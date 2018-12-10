import os
import argparse
from shutil import copytree
import fnmatch
import pandas as pd
from shutil import copyfile
from colorsutils import bcolors
from tqdm import tqdm
import os

dict_renaming = {
    "*DE*SPC_30*Std*" : "SPC_301mm_Std",
    "*T2W*": "t2_tse_tra",
    "*TRACEW*": "diff_trace_tra_TRACEW",
    "*ADC*": "diff_trace_tra_ADC"
}

DEBUG = False

def create_tree_path(folder, pattern):

    filters = pattern.split("\\")
    filters = [f for f in filters if len(f.replace("*","")) > 0]
    folders = folder.split("\\")

    selected_folders = []
    for ft in filters:
        for f in folders:
            if fnmatch.fnmatch(f, ft):
                if f not in selected_folders:
                    selected_folders.append(f)
    if len(selected_folders)<1:
        print(bcolors.FAIL+"COULD COPY : {}-- replace with t2_tse_tra".format(folder)+bcolors.ENDC)
        folder = folder.replace("T2W_TSE_tra", "t2_tse_tra")
        return folder
    else:
        entire_path = os.path.join(*selected_folders)
    return entire_path


def correct_folder_name(dir, dict_renaming):
    folder_name = os.path.basename(dir)
    for p, d in dict_renaming.items():
        if fnmatch.fnmatch(folder_name.lower(), p.lower()):
            new_name = dir.replace(folder_name, d)
            print_message(message=bcolors.BOLD,end="") # To print in BOLD in command : see that the folder was renamed.
            return new_name
    return dir


def print_message(message,end="END"):
    global DEBUG
    end = "END"
    if DEBUG:
        if len(end)>0:
            print(message)
        else:
            print(message,end="")

def filter_and_copy(main_folder, patterns, output_dir, skipped=None):
    folder = main_folder

    for dirs, subdirs, files in os.walk(folder):
        if("d:\." in dirs):
            continue
        for p in patterns:
            if fnmatch.fnmatch(dirs, p) or fnmatch.fnmatch(dirs.lower(), p.lower()):
                new_path = os.path.basename(create_tree_path(dirs, p))
                output_path = os.path.join(output_dir, new_path)
                output_path = correct_folder_name(output_path, dict_renaming)
                print_message(dirs+ "->"+ output_path)
                print_message(message=bcolors.ENDC,end="")
                if os.path.exists(output_path):
                    print_message(bcolors.WARNING+"Already exists : {}".format(output_path)+bcolors.ENDC)
                else:
                    copytree(dirs, output_path)
            for filename in files:
                filename = os.path.join(dirs,filename)
                if fnmatch.fnmatch(filename,p) and fnmatch.fnmatch(filename, "*VOI*.nii"):
                    new_path = os.path.basename(create_tree_path(filename, p))
                    output_path = os.path.join(output_dir, new_path)
                    print_message(filename+ "->"+ output_path)
                    if os.path.exists(output_path):
                        print_message(bcolors.WARNING+"Already exists : {}".format(output_path)+bcolors.ENDC)
                    else:
                        copyfile(filename, output_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Filter and copy files.')
    parser.add_argument("-p", "--path", type=str, help="Main directory path", default="/media/exfat")
    parser.add_argument("-f", "--file", type=str, help="Path to conversion file", default="patient_mappings.csv")
    parser.add_argument("-o", "--output_dir", type=str, help="Output root directory", default="/media/exfat/DataFilter")
    parser.add_argument("-d", "--debug", type=str, help="Debug and print copy", default=False)
    parser.add_argument("-s", "--skip_empty_lesion", type=str, help="Skip patient if lesion is absent", default=True)

    args = parser.parse_args()
    mappings = args.file
    root_folder = args.path
    output_dir = args.output_dir
    debug = args.debug
    skip_empty_lesion = args.skip_empty_lesion

    df = pd.read_csv(mappings, header=None)
    skipped = []
    rows = [(idx,row) for idx,row in df.iterrows()]
    for id,row in tqdm(rows):
        root_dir = row[0].split("/")[1]

        splitted = [x for x in row[0].split("\\")]
        patterns = [os.path.join(*splitted)]
        outdir = os.path.join(*[x for x in row[1].split("\\")])
        out_folder = os.path.join(output_dir, outdir)
        in_folder = os.path.join(root_folder, root_dir)

        # Get patient folder and test if lesion exists
        rooted_folder = in_folder.replace(root_folder, "").replace("/","")
        patient_folder = os.path.join(root_folder, rooted_folder)
        files_in_patient_folder = os.listdir(patient_folder)


        if not any([os.path.basename(f).startswith("VOI") for f in files_in_patient_folder]) \
                and not (patient_folder in skipped):
            print(bcolors.FAIL + "NO LESION FOUND FOR " + patient_folder + bcolors.ENDC)
            skipped.append(patient_folder)
            continue

        # Skip invalid patients
        if (patient_folder in skipped) and skip_empty_lesion:
            print_message("Skipping patient " + patient_folder)
            continue

        filter_and_copy(in_folder, patterns, out_folder)
