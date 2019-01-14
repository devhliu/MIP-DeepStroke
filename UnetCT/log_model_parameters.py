from argparse import ArgumentParser
import os
import json
import pandas as pd

if __name__ == '__main__':
    parser = ArgumentParser(description="Evaluates a 3D Unet model")
    parser.add_argument("-l", "--logdir", help="Directory where the Keras models are stored",
                        default="/home/snarduzz/Models")
    parser.add_argument("-o", "--output_file", help="Directory where the Keras models are stored",
                        default="/home/snarduzz/Models/models_parameters.csv")

    args = parser.parse_args()
    models = [os.path.join(args.logdir, x) for x in os.listdir(args.logdir)]
    models = [x for x in models if os.path.isdir(x) ]

    series = []
    for model in sorted(models):
        files = os.listdir(model)
        if "parameters.json" not in files:
            print("No parameters found in {}".format(model))
            continue

        json_file = os.path.join(model, "parameters.json")

        with open(json_file) as f:
            data = json.load(f)
        data["date"] = os.path.basename(model)
        serie = pd.Series(data)
        series.append(serie)

    df = pd.DataFrame(series)
    df.to_csv(args.output_file)