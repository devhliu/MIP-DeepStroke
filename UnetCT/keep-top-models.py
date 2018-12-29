from argparse import ArgumentParser, REMAINDER
import os
from tqdm import tqdm


def parseLoss(model_name):
    split = model_name.split("-")
    if len(split)>2:
        score = split[2].replace(".hdf5","")
    else:
        score = split[1].replace(".hdf5", "")
    score = float(score)
    if score < 0:
        score = -score
    return score


def parseIteration(model_name):
    iteration = model_name.split("-")[0].replace("model.","")
    return iteration


def remove(file):
    os.remove(file)


def keep_checkpoints(path):
    # list all checkpoints files
    files = os.listdir(path)
    files = sorted(files, key=parseLoss, reverse=True)
    keeped = [x for x in files if parseIteration(x) in exception_iteration]

    all_files = set(files[:top] + keeped)
    for f in tqdm(files):
        if f not in all_files:
            remove(os.path.join(path, f))

    print("File keeped : ")
    for f in all_files:
        print(f)


if __name__ == '__main__':
    parser = ArgumentParser(description="Keep the best models in folder")
    parser.add_argument("-l", "--logdir", help="Directory where the Keras models are stored",
                        default="/home/snarduzz/Models")

    parser.add_argument("-t", "--top", help="Number of top models", type=int, default=2)
    parser.add_argument('-k', '--keep', nargs="*", help='Iteration to keep. Usage : -k 678 754 23 to keep iterations "678","754" and "23" ')

    args = parser.parse_args()
    exception_iteration = set()
    if args.keep:
        exception_iteration = set([x for x in args.keep])
        print(exception_iteration)
    logdir = os.path.expanduser(args.logdir)
    top = args.top

    folders = os.listdir(logdir)
    if "checkpoints" not in folders:
        # list models
        models = os.listdir(logdir)
        for m in models:
            path = os.path.join(logdir, m, "checkpoints")
            if not os.path.exists(path):
                raise Exception("No checkpoints folder found in {}".format(folders))
    else:
        path = os.path.join(logdir, "checkpoints")
        if not os.path.exists(path):
            raise Exception("No checkpoints folder found in {}".format(folders))
        keep_checkpoints(path)


