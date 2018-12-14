import os
import shutil
from tqdm import tqdm

main_dir = "/media/exfat/stroke_db/2016/2016_without_VOI (copy)"
copy_main_dir = "/media/exfat/stroke_db/2016/2016_without_VOI_clean"

if not os.path.exists(copy_main_dir):
    os.mkdir(copy_main_dir)

patients = [os.path.join(x) for x in os.listdir(main_dir) if not x.startswith("._")]

for p in tqdm(patients):
    new_folder = p.replace(" ","_").replace(","," ")
    new_folder_tree = os.path.join(copy_main_dir,new_folder)
    old_folder_tree = os.path.join(main_dir, p)
    os.rename(old_folder_tree, new_folder_tree)