from argparse import ArgumentParser
from utils import create_if_not_exists
from image_processing import transform_atlas_to_patches, splits_sets
import os


if __name__ == '__main__':
    parser = ArgumentParser(description="Train a 3D Model unet")
    parser.add_argument("-l", "--logdir", help="Directory where to log the data for tensorboard",
                        default="/home/snarduzz/")

    parser.add_argument("-d", "--data_path", help="Path to data folder (ATLAS)",
                        default="/home/simon/Datasets/Stroke_DeepLearning_ATLASdataset")
    parser.add_argument("-s", "--save_path", help="Path where to save patches", default="/home/simon/Datasets")
    args = parser.parse_args()

    patch_size = [32, 32, 32]
    string_patches = "x".join([str(x) for x in patch_size])

    dataset_path = create_if_not_exists(args.save_path)
    dataset_data_path = create_if_not_exists(os.path.join(dataset_path, "Data"))
    save_path = create_if_not_exists(os.path.join(dataset_data_path, string_patches))

    #patches_path = transform_atlas_to_patches(altas_path=args.data_path, save_path=save_path, patch_size=patch_size)

    #print("Patches saved under {}".format(patches_path))
    splits_sets(save_path, ratios=[0.6, 0.15, 0.25])
    print("Sets saved under {}".format(save_path))