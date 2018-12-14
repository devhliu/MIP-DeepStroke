import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter and copy files.')
    parser.add_argument("-p", "--path", type=str, help="Main directory path", default="/media/exfat/DataFilter")
    parser.add_argument("-f", "--files", type=str, help="Folder should contain", default=None)

    args = parser.parse_args()

    path = args.path
    #folders = args.files
    folders = ["Ct2_Cerebrale",
               "Ct2_Cerebrale/RAPID_MTT",
               "Ct2_Cerebrale/RAPID_Tmax",
               "Ct2_Cerebrale/RAPID_rCBV",
               "Ct2_Cerebrale/RAPID_rCBF",
               "Ct2_Cerebrale/SPC_301mm_Std",
               "Neuro_Cerebrale_64Ch",
               "Neuro_Cerebrale_64Ch/diff_trace_tra_ADC",
               "Neuro_Cerebrale_64Ch/diff_trace_tra_TRACEW",
               "Neuro_Cerebrale_64Ch/t2_tse_tra",
               "VOI lesion.nii",
               ]

    patients = os.listdir(path)
    i = 0

    for p in patients:
        missing = []
        patient_folder = os.path.join(path, p)
        for f in folders:
            folder = os.path.join(patient_folder, f)
            if not os.path.exists(folder):
                missing.append(f)
        if len(missing) > 0:
            i = i+1
            print("[{}] Missing : {}".format(p, missing))
    print("\n{} Patient failed.".format(i))
