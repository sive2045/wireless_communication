import multiprocessing
from multiprocessing.dummy import freeze_support

num_cores = multiprocessing.cpu_count()

def squreas(input_list):
    return [x*x for x in input_list]

import numpy as np

data = list(range(1,25))
splited_data = np.array_split(data, num_cores)
splited_data = [x.tolist() for x in splited_data]

import parmap

if __name__ == '__main__':
    freeze_support()
    result = parmap.map(squreas, splited_data, pm_pbar=True, pm_processes=num_cores)
    print(result)