import os
import shutil
from tqdm import tqdm

main_dir = "/home/snarduzz/Data/Data_2016_T2_TRACE_LESION"
#output_dir = "/media/exfat/stroke_db/2016/2016_without_VOI_copy_py"

patients = [os.path.join(main_dir,x) for x in os.listdir(main_dir) if not x.startswith("._")]

i = 0
for p in tqdm(patients):
    folders = [f for f in os.listdir(p) if not f.startswith(".")]
    if len(folders)<2:
        continue
        print(folders)
    else:
        print(os.path.basename(p), [x for x in folders if "ct" in x.lower() or "neuro" in x.lower()])
