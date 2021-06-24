# ---------- imports ----------- #
import Fidelity_calculator
import os
import numpy as np
from matplotlib import pyplot as plt

# ----------- macros ----------- #
ANCILLAS = 1
TRANS = 2


# ------------ code ------------ #

def img_from_csv(path, changing):
    """

    :param changing: string indicating the changes parameter - #ancilla or #tranparencies
    :param path: relative path to folder of csv files which represent images
    or a single csv file. handles both cases
    :return:
    """
    img_arr = []
    parent_dir = path.split("/")[-1]
    if os.path.isdir(path):  # handle directory

        only_files = [os.path.abspath(os.path.dirname(f)) + "\\" + parent_dir + "\\" + f
                      for f in os.listdir(path) if
                      os.path.splitext(os.path.abspath(f))[1] == ".csv"]  # filter .csv files
        img_arr = [(np.abs(np.genfromtxt(file, delimiter=',', dtype=None, encoding=None)), get_name(file)) for file in
                   only_files]  # converts csv to np array

    elif os.path.isfile(path):  # handle single file
        name, extension = os.path.splitext(path)
        if extension != ".csv":
            print("Given path doesn't contain a .csv file")
            return
        img_arr.append(np.abs(np.genfromtxt(path, delimiter=",", dtype=None, encoding=None)))
    else:
        print("Given path is not a Directory or a file")
    return img_arr


def plot_err(orig_img_path, path, err_func):
    pass


def get_name(file_path, changing):
    """
    returns the name of the file with number of ancillas/ transparencies used.
    :param changing:
    :param file_path: absolute path to given file
    :return: string of the format - "# ancillas/transparencies"
    """
    name = file_path.split("\\")[-1].split(",")[1]
    if changing == ANCILLAS:
        return name
    elif changing == TRANS:
        return name.split('.')[0]  # removes .csv


if __name__ == '__main__':
    images = img_from_csv("C:/Users/user/PycharmProjects/FidelityProject/change_ancillas_1", "blah")
    for img in images:
        plt.imshow(img[0], cmap='gray')
        plt.title(img[1])
        plt.show()
