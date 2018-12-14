import os
import nibabel


main_dir = "/home/snarduzz/Data/Data_2016_T2_TRACE_LESION"
patients = os.listdir(main_dir)

for p in patients:
    patient_dir= os.path.join(main_dir, p)
    folders = [x for x in os.listdir(patient_dir) if ".nii" not in x]
    for folder in folders:
        files = os.listdir(os.path.join(patient_dir,folder))

        for f in files:
            if "trace" in f.lower():
                img = nibabel.load(os.path.join(patient_dir,folder,f)).get_data()

                if(len(img.shape)<4):
                    print("ONLY ONE SCAN : ",patient_dir)

                elif(img.shape[3]!=2):
                    print("MORE DIMENSIONS {} : ".format(img.shape[3]), patient_dir)