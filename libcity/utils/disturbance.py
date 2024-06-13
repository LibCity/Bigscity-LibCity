import numpy as np


def get_disturb_indices(data, rate):
    shape = data.shape
    total_elements = np.prod(shape[:-1])
    num_elements_to_noise = int(total_elements * rate)
    random_indices = np.random.choice(total_elements, num_elements_to_noise, replace=False)
    indices = np.unravel_index(random_indices, shape[:-1])
    return indices, num_elements_to_noise


def zero_noise(data, rate, dim=None):
    if dim is None:
        dim = data.shape[-1]
    for i in range(dim):
        indices, num_elements_to_noise = get_disturb_indices(data, rate)
        data[indices + (i, )] = 0
    return data


def gaussian_noise(data, rate, mean, std, dim=None):
    if dim is None:
        dim = data.shape[-1]
    assert (len(mean) == dim), 'The number of data features is different from the number of Gaussian distributions.'
    for i in range(dim):
        indices, num_elements_to_noise = get_disturb_indices(data, rate)
        noise = np.random.normal(loc=mean[i], scale=std[i], size=num_elements_to_noise)
        data[indices + (i, )] += noise
    return data


if __name__ == '__main__':
    data = np.random.random((2, 2, 2, 3))
    gaussian_noise(data, 0.4, [1,2], [2,1], 2)
