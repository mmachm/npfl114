import numpy as np
from collections import defaultdict
from math import inf

if __name__ == "__main__":
    with open("numpy_entropy_data.txt", "r") as data:
        distr_data_model = defaultdict(lambda: [0, 0])
        n = 0
        for line in data:
            n += 1
            line = line.rstrip("\n")
            distr_data_model[line][0] += 1
        if n:
            for key in distr_data_model:
                distr_data_model[key][0] /= n
    
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            key, p = line.rstrip("\n").split("\t")
            p = float(p)
            distr_data_model[key][1] = p
    data_list, model_list = [], []
    for key, val in distr_data_model.items():
        data_list.append(val[0])
        model_list.append(val[1])
        blowup = False
        if val[0] and not val[1]:
            blowup = True
    model_array = np.array(model_list)
    data_array = np.array(data_list)
    print(model_array, data_array)
    print(blowup, distr_data_model) 
    entropy = -np.sum(data_array * np.ma.log(data_array).filled(0))
    if blowup:
        cross_entropy = inf
    else:
        cross_entropy = -np.sum(data_array * np.log(model_array))
    print("{:.2f}".format(entropy))
    print("{:.2f}".format(cross_entropy))
    print("{:.2f}".format(cross_entropy-entropy))

