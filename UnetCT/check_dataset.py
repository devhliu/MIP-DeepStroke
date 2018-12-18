import numpy as np
import os
from tqdm import tqdm
import nibabel as nb

from argparse import ArgumentParser
import json

if __name__ == '__main__':

    parser = ArgumentParser(description="Train a 3D Model unet")
    parser.add_argument("-p", "--path", help="Directory where to log the data for tensorboard",
                        default="/home/snarduzz/Data")

    args = parser.parse_args()
    directory = args.path

    folders = os.listdir(directory)
    folders = [x for x in folders if x in ["train","test","validation"]]

    dict_log = dict()


    if len(folders)==0:
        raise Exception("The root directory should contain train/test/validation sets")

    for f in folders:
        subdirectory = os.path.join(directory,f)
        channels = os.listdir(subdirectory)

        total_containing_lesion = 0
        total_not_containing_lesion = 0
        total_lesion_files = 0

        total_lesion_voxels = 0
        total_not_lesion_voxels= 0
        total_voxels = 0

        for c in channels:
            subsubdir = os.path.join(subdirectory, c)
            files = os.listdir(subsubdir)
            files = [x for x in files if x.endswith(".nii")]
            print("Found {} .nii files. Checking....".format(len(files)))

            for file in tqdm(files, desc=str(c)):
                file = os.path.join(subsubdir, file)
                img = nb.load(file).get_data()
                max_img = np.max(img)
                min_img = np.min(img)
                if max_img>1 or min_img<-1:
                    dict_log[file] = (min_img, max_img)

                if c.lower() == "lesion":
                #Log number of files with lesion
                    if np.max(img)>=1:
                        total_containing_lesion+=1
                #Log number of files without lesion
                    if np.max(img)<1:
                        total_not_containing_lesion+=1
                    total_lesion_files+=1

                    total_lesion_voxels += np.sum(img)
                    total_voxels_img = img.shape[0]*img.shape[1]*img.shape[2]
                    total_not_lesion_voxels += (total_voxels_img-np.sum(img))
                    total_voxels += total_voxels_img


        dict_log["{}_CONTAINING_LESION".format(f)] = total_containing_lesion
        dict_log["{}_NOT_CONTAINING_LESION".format(f)] = total_not_containing_lesion
        dict_log["{}_TOTAL_FILES".format(f)] = total_lesion_files
        dict_log["{}_VOXELS_CONTAINING_LESION".format(f)] = total_lesion_voxels
        dict_log["{}_VOXELS_NOT_CONTAINING_LESION".format(f)] = total_not_lesion_voxels
        dict_log["{}TOTAL_VOXELS".format(f)] = total_voxels

    # Save report
    json_file = os.path.join(directory, "report.json")
    with open(json_file, 'w') as fp:
        json.dump(dict_log, fp)

    print("Found {} incorrect files. Report is available here : {}".format(len(dict_log)-3*len(folders), json_file))


    total_dataset_size = np.array([dict_log["{}_TOTAL_FILES".format(f)] for f in folders]).sum()

    print("\nDataset description----------")
    for f in folders:
        total_lesion = dict_log["{}_CONTAINING_LESION".format(f)]
        total_not_lesion = dict_log["{}_NOT_CONTAINING_LESION".format(f)]
        total = dict_log["{}_TOTAL_FILES".format(f)]
        total_lesion_voxels =  dict_log["{}_VOXELS_CONTAINING_LESION".format(f)]
        total_not_lesion_voxels =  dict_log["{}_VOXELS_NOT_CONTAINING_LESION".format(f)]
        total_voxels = dict_log["{}TOTAL_VOXELS".format(f)]

        print("--- {} ".format(f.upper()))
        print("Lesions files: {} ({}%)".format(str(total_lesion), str(100*total_lesion/total)))
        print("Not lesions files: {} ({}%)".format(str(total_not_lesion), str(100*total_not_lesion / total)))
        print("TOTAL : {} ({}% of total dataset)".format(str(total), str(100*total/total_dataset_size)))
        print("")

        print("--- {} ".format(f.upper()))
        print("Lesions voxels: {} ({}%)".format(str(total_lesion_voxels), str(100 * total_lesion_voxels / total_voxels)))
        print("Not lesions voxels: {} ({}%)".format(str(total_not_lesion_voxels), str(100 * total_not_lesion_voxels / total_voxels)))
        print("Total voxels : {}".format(str(total_voxels)))
        print("")


    print("Total dataset size : {} files.".format(total_dataset_size))



