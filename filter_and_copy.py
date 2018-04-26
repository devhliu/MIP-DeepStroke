import os
import argparse
from distutils.dir_util import copy_tree
import fnmatch
import pandas as pd
from shutil import copyfile
from utils import bcolors


dict_renaming = {
    "*DE*SPC_30*Std*" : "SPC_301mm_Std",
    "T2W*": "t2_tse_tra",
}

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
    entire_path = "\\".join(selected_folders)
    return entire_path


def correct_folder_name(dir, dict_renaming):
    folder_name = os.path.basename(dir)
    for p, d in dict_renaming.items():
        if fnmatch.fnmatch(folder_name.lower(), p.lower()):
            new_name = dir.replace(folder_name, d)
            print(bcolors.BOLD,end="") # To print in BOLD in command : see that the folder was renamed.
            return new_name
    return dir


def filter_and_copy(main_folder, patterns, output_dir):
    folder = main_folder
    for dirs, subdirs, files in os.walk(folder):
        if("d:\." in dirs):
            continue
        for p in patterns:
            if fnmatch.fnmatch(dirs, p) or fnmatch.fnmatch(dirs.lower(), p.lower()):
                new_path = os.path.basename(create_tree_path(dirs, p))
                output_path = os.path.join(output_dir, new_path)
                output_path = correct_folder_name(output_path, dict_renaming)
                print(dirs, "->", output_path)
                print(bcolors.ENDC,end="")
                if os.path.exists(output_path):
                    print(bcolors.WARNING+"Already exists : {}".format(output_path)+bcolors.ENDC)
                else:
                    copy_tree(dirs,output_path)
            for filename in files:
                filename = os.path.join(dirs,filename)
                if fnmatch.fnmatch(filename,p) and fnmatch.fnmatch(filename,"*VOI*.nii"):
                    new_path = os.path.basename(create_tree_path(filename, p))
                    output_path = os.path.join(output_dir, new_path)
                    print(filename, "->", output_path)
                    if os.path.exists(output_path):
                        print(bcolors.WARNING+"Already exists : {}".format(output_path)+bcolors.ENDC)
                    else:
                        copyfile(filename, output_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Filter and copy files.')
    parser.add_argument("-p", "--path", type=str, help="Main directory path")
    parser.add_argument("-f", "--file", type=str, help="Path to conversion file")
    parser.add_argument("-o", "--output_dir", type=str, help="Output root directory")

    args = parser.parse_args()
    mappings = args.file
    root_folder = args.path
    output_dir = args.output_dir

    df = pd.read_csv(mappings,header=None)
    for id,row in df.iterrows():
        root_dir = row[0].split("\\")[1]
        patterns = [row[0]]
        out_folder = os.path.join(output_dir, row[1])
        in_folder = os.path.join(root_folder, root_dir)
        filter_and_copy(in_folder, patterns, out_folder)
