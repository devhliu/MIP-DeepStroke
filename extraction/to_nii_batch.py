import os
from shutil import copyfile
import subprocess

# mount = '/run/media/jk/Elements'
# main_dir = os.path.join(mount, 'MASTER/')
main_dir = '/media/exfat/stroke_db/2016'
data_dir = os.path.join(main_dir, 'DATA_T2_TRACE_LESION')
output_dir = os.path.join(main_dir, 'Data_2016_T2_TRACE_LESION')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
dcm2niix_path = "/usr/bin/dcm2niix"

subjects = os.listdir(data_dir)
print(subjects)

i=0
for subject in list(subjects):
    i=i+1
    print("--------------------------------------{}/{}".format(i,len(subjects)))
    subject_dir = os.path.join(data_dir, subject)
    # copy lesion
    if not os.path.isdir(subject_dir):
        continue
    else:
        subject_output_dir = os.path.join(output_dir,subject)
        if not os.path.exists(subject_output_dir):
            os.mkdir(subject_output_dir)

    modalities = [o for o in os.listdir(subject_dir)
                    if os.path.isdir(os.path.join(subject_dir,o))]
    files = [o for o in os.listdir(subject_dir)
                    if os.path.isfile(os.path.join(subject_dir,o))]
    for f in files:
        inputfile = os.path.join(data_dir,subject,f)
        outputfile = os.path.join(output_dir,subject,f)
        copyfile(inputfile,outputfile)

    for modality in modalities:
        modality_dir = os.path.join(subject_dir, modality)
        studies = [o for o in os.listdir(modality_dir)
                        if os.path.isdir(os.path.join(modality_dir,o))]


        for study in studies:
            study_dir = os.path.join(modality_dir, study)
            study_output_dir = os.path.join(output_dir, subject, modality, study)
            if not os.path.exists(study_output_dir):
                os.makedirs(study_output_dir)
            subprocess.run([dcm2niix_path, '-m', 'y', '-o', study_output_dir, study_dir], cwd = modality_dir)