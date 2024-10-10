import numpy as np


def max_fn(x):
    x_max = np.where(x > 0, x, 0)
    return x_max / np.sum(x_max)


def get_sample(p):
    # here p is given bc we wanna allocate the same probability to each token,
    # if p=[.25, .25] then uniform else, higher prob will go to higher token
    # print(p)
    # print(np.arange(p.shape[-1]))
    return np.random.choice(np.arange(p.shape[-1]), p=p)


# used np.array for the broadcasting feature of the numpy n elementwise operation
# print(max_fn(np.array([1,2,-2,5,5])))
# print(get_sample(np.array([0.1,0.3, 0.6])))
