import os
import subprocess
from multiprocessing import Pool
from argparse import ArgumentParser
from tqdm import tqdm
from shutil import copy, rmtree, copyfile
import itertools

# mount = '/run/media/jk/Elements'
# main_dir = os.path.join(mount, 'MASTER/')
main_dir = "/home/simon/Documents/EPFL/Master/SemesterProject/Data"
data_dir = os.path.join(main_dir, 'Anonymized_Data')
output_dir = os.path.join(main_dir, 'To_Preprocess')
dcm2niix_path = "/usr/bin/dcm2niix"

ct_sequences = ['SPC_301mm_Std', 'RAPID_Tmax', 'RAPID_MTT', 'RAPID_rCBV', 'RAPID_rCBF']
mri_sequences = ['t2_tse_tra', 'T2W_TSE_tra']
sequences = ct_sequences + mri_sequences


def create_if_not_exists(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def organize_folder(subject_folder, output_dir):
    subject = os.path.basename(subject_folder)
    if os.path.isdir(subject_folder):
        modalities = [o for o in os.listdir(subject_folder) if os.path.isdir(os.path.join(subject_folder, o))]

        for modality in modalities:
            modality_dir = os.path.join(subject_folder, modality)
            modality_output_dir = os.path.join(output_dir, subject, modality)

            if not os.path.exists(modality_output_dir):
                os.makedirs(modality_output_dir)

            studies = [o for o in os.listdir(modality_dir) if os.path.isdir(os.path.join(modality_dir, o))]

            for study in studies:
                study_dir = os.path.join(modality_dir, study)
                if study in sequences:
                    for file in os.listdir(study_dir):
                        if file.endswith(".nii"):
                            file_path = os.path.join(study_dir, file)
                            new_file_name = study + '_' + subject + '.nii'
                            new_file_path = os.path.join(modality_output_dir, new_file_name)
                            if not os.path.exists(new_file_path):
                                copy(file_path, new_file_path)

        # copy lesions file into subject dir
        lesion_path = os.path.join(data_dir, subject, 'VOI lesion.nii')
        new_lesion_name = 'VOI_lesion_' + subject + '.nii'
        storage_path = os.path.join(output_dir, subject)
        create_if_not_exists(storage_path)
        new_lesion_path = os.path.join(storage_path, new_lesion_name)
        if not os.path.exists(new_lesion_path):
            # print(new_lesion_path)
            copyfile(lesion_path, new_lesion_path)


def to_nii(subject_folder, output_dir):
    subject = os.path.basename(subject_folder)
    modalities = [o for o in os.listdir(subject_folder) if os.path.isdir(os.path.join(subject_folder, o))]

    for modality in modalities:
        modality_dir = os.path.join(subject_folder, modality)
        studies = [o for o in os.listdir(modality_dir) if os.path.isdir(os.path.join(modality_dir, o))]

        for study in studies:
            study_dir = os.path.join(modality_dir, study)
            study_output_dir = os.path.join(output_dir, subject, modality, study)

            if not os.path.exists(study_output_dir):
                os.makedirs(study_output_dir)

            subprocess.run([dcm2niix_path, '-m', 'y', '-o', study_output_dir, study_dir], cwd=data_dir)


def skull_strip(folder, skull_strip_path):
    modalities = [o for o in os.listdir(folder)
                  if os.path.isdir(os.path.join(folder, o))]

    for modality in modalities:
        modality_dir = os.path.join(folder, modality)
        studies = [o for o in os.listdir(modality_dir)
                   if os.path.isfile(os.path.join(modality_dir, o))]

        for study in studies:
            if modality.startswith('Ct') & study.startswith('SPC'):
                # BET image
                subprocess.run([skull_strip_path, '-i', study],
                                        cwd=os.path.join(folder, modality))

                # Extract file (*nii.gz created by fslmaths)
                created_files = [x for x in os.listdir(modality_dir) if
                                 x.startswith("betted_") and x.endswith(".nii.gz")]
                for file in created_files:
                    subprocess.run(["gzip", "-d", file], cwd=os.path.join(folder, modality))


def process_folder(subject_folder, output_dir, tmp_folder=None):
    subject = os.path.basename(subject_folder)

    if tmp_folder is None:
        tmp_folder = os.path.join("/tmp/Temp_Nii")
        create_if_not_exists(tmp_folder)

    to_nii(subject_folder, tmp_folder)

    # Copy lesion
    files = [file for file in os.listdir(subject_folder) if file.endswith(".nii") and "lesion" in file.lower()]
    for file in files:
        dest = os.path.join(tmp_folder, subject)
        create_if_not_exists(dest)
        copyfile(os.path.join(subject_folder, file), os.path.join(tmp_folder, subject, file))

    # Reorganize
    print("Reorganizing : {}".format(subject))
    subject_folder = os.path.join(tmp_folder, subject)
    create_if_not_exists(output_dir)
    organize_folder(subject_folder, output_dir)

    rmtree(os.path.join(tmp_folder, subject))


if __name__ == '__main__':
    parser = ArgumentParser("Convert dcm images to nii images")
    parser.add_argument("-p", "--path", help="Path to the data containing the subjects", default=data_dir)
    parser.add_argument("-o", "--output", help="Output directory", default=output_dir)
    parser.add_argument("-d", "--dcm2niix", help="Path to the dcm2niix executable", default=dcm2niix_path)
    parser.add_argument("-m", "--multiprocessing", help="Number of threads used to multiprocess the data", default=4)

    args = parser.parse_args()

    data_dir = args.path
    output_dir = args.output
    dcm2niix_path = args.dcm2niix
    processes = args.multiprocessing

    subjects_folders = [os.path.join(args.path, x) for x in os.listdir(args.path)]

    base_name_data_dir = os.path.basename(data_dir)
    tmp_folder = data_dir.replace(base_name_data_dir, "Temp_Nii")
    create_if_not_exists(tmp_folder)


    print("--------Converting files to nii, copy lesion and organize-----------")
    # Create a pool of workers
    pool = Pool(processes)
    pool.starmap(process_folder,  zip(subjects_folders,
                                      itertools.repeat(output_dir, len(subjects_folders)),
                                      itertools.repeat(tmp_folder, len(subjects_folders))))
    pool.close()
    pool.join()

    # Cleanup
    rmtree(tmp_folder)
 
    print("---------Betting-----------------------------")
    skull_strip_path = os.path.join(os.getcwd(), 'skull_strip.sh')
    subjects = [os.path.join(output_dir, x) for x in os.listdir(output_dir)]
    skull_strip_paths = itertools.repeat(skull_strip_path, len(subjects))

    # Create Pool of workers
    pool = Pool(processes)
    pool.starmap(skull_strip, zip(subjects, skull_strip_paths))
    pool.close()
    pool.join()
