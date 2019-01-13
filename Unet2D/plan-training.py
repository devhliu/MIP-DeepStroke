import os
import json
import subprocess as sub


custom_parameter_file = "/home/snarduzz/parameters-iterations.json"
parameters_tmp = "/home/snarduzz/parameters-tmp.json"

print("Loading parameters from : " + custom_parameter_file)
with open(custom_parameter_file, 'r') as fp:
    custom_base_dict = json.load(fp)

dict_iteration = custom_base_dict.copy()

alpha_values = [0.002, 0.0038, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
architectures = "unet"  #["unet", "attn_reg", "attn_reg_ds", "attn_unet"]
augmentation = 0.5  # [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
loss_functions = "tversky" # ["tversky", "jaccard", "focal_tversky"]

iterate_on = "tversky_alpha-beta"

dict_parameters = dict()
dict_parameters["tversky_alpha-beta"] = alpha_values
dict_parameters["architecture"] = architectures
dict_parameters["loss_function"] = loss_functions
dict_parameters["augmentation"] = augmentation

for value in dict_parameters[iterate_on]:

    if iterate_on == "tversky_alpha-beta":
        new_value = [value, 1-value]
    else:
        new_value = value

    dict_iteration[iterate_on] = new_value

    print("NEW PARAMETERS : {}".format(dict_iteration))
    with open(parameters_tmp, 'w') as fp:
        json.dump(dict_iteration, fp, indent=4)

    os.chdir("/home/snarduzz/MIP-DeepStroke/Unet2D")
    command = "python3 train_2D.py -i TRACE -o LESION -params /home/snarduzz/parameters-tmp.json"
    p = os.popen(command).read()
