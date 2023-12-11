import glob
import os
import json
import numpy as np

if __name__ == '__main__':
    os.chdir("results")

    results = {'original reward': [],
               'hinge 0.1 reward': [],
               'hinge 1.0 reward': [],
               'sigmoid 0.1 reward': [],
               'sigmoid 1.0 reward': [],
               'original div': [],
               'hinge 0.1 div': [],
               'hinge 1.0 div': [],
               'sigmoid 0.1 div': [],
               'sigmoid 1.0 div': [],
               'hinge 0.1 std': [],
               'hinge 1.0 std': [],
               'sigmoid 0.1 std': [],
               'sigmoid 1.0 std': [],
               }

    names = ['hinge 0.1', 'hinge 1.0', 'sigmoid 0.1', 'sigmoid 1.0']
    for pattern in names:
        names = glob.glob(f"{pattern}*")
        for name in names:

            with open(name) as f:
                data = json.load(f)
                results[f"{pattern} reward"].append(data['model_ft avg reward'])
                results[f"{pattern} div"].append(data['model_ft diversity'])
                results[f"{pattern} std"].append(data['model_ft std reward'])
                results[f"original reward"].append(data['model avg reward'])
                results[f"original div"].append(data['model diversity'])

    for key, value in results.items():
        print(f"{key} mean: {np.mean(value)}, std: {np.std(value)}")

