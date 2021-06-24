# ---------- imports ----------- #
import Fidelity_calculator as fc
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgba2rgb, rgb2gray

# ----------- macros ----------- #
ANCILLAS = 1
TRANS = 2


# ------------ code ------------ #

def img_from_csv(path, parameter):
    """
    reads image if a file path is given, or reads multiple csv files if a directory is given.
    :param parameter: int representing what is the changing parameter - 1 for ancillas, 2 for transparencies.
    :param path: relative path to folder of csv files which represent images
    or a single csv file. handles both cases
    :return: python list of tuples in the format (image, # of ancillas/transparencies) where image is np.array.
    """
    ret = []
    parent_dir = path.split("/")[-1]
    if os.path.isdir(path):  # handle directory

        only_files = [os.path.abspath(os.path.dirname(f)) + "\\" + parent_dir + "\\" + f
                      for f in os.listdir(path) if
                      os.path.splitext(os.path.abspath(f))[1] == ".csv"]  # filter .csv files
        ret = [(np.abs(np.genfromtxt(file, delimiter=',', dtype=None, encoding=None)), get_name(file, parameter)) for
               file in
               only_files]  # converts csv to np array

    elif os.path.isfile(path):  # handle single file
        name, extension = os.path.splitext(path)
        if extension != ".csv":
            print("Given path doesn't contain a .csv file")
            return
        ret.append((np.abs(np.genfromtxt(path, delimiter=",", dtype=None, encoding=None)), get_name(path, parameter)))
    else:
        print("Given path is not a Directory or a file")
    return sorted(ret, key=lambda x: int(x[1]))


def plot_err(orig_img_path, path, param, err_func="ALL"):
    """

    :param orig_img_path: string - path to original image
    :param path: string - path to directory of images for comparison
    :param err_func: string - indicating which error function to use, default is all.
    :param param: int - indicating if the changing parameter is acillas or transparencies
    :return:
    """
    err_mat = None
    orig_img = fc.read_image(orig_img_path)
    to_compare_lst = img_from_csv(path, param)
    # creates matrix where each column is a the error according to diff func
    for tup in to_compare_lst:
        curr_img = rgb2gray(tup[0].astype(np.float32))
        curr_img = (curr_img / np.max(curr_img)) * 255  # normalize image
        curr_err = fc.compare_img(orig_img, curr_img, err_func)  # get errors according to diff error functions
        if err_mat is not None:
            err_mat = np.vstack((err_mat, curr_err))
        else:
            err_mat = curr_err

    # plot the error according to the changing parameter
    x_arr = np.array([int(tup[1]) for tup in to_compare_lst])
    titles = ["MSE", "SSIM", "L1 Norm"]
    for i in range(err_mat.shape[1]):
        plt.scatter(x_arr, err_mat.T[i])
        plt.plot(x_arr, err_mat.T[i], c="navy")
        plt.xticks(x_arr)
        if param == ANCILLAS:
            plt.xlabel("# of ancillas")
            temp = "Ancillas"
        else:
            plt.xlabel("# of transparencies")
            temp = "Transparencies"
        plt.ylabel("error")
        plt.title(titles[i] + " vs. # of " + temp)
        plt.savefig(titles[i])
        plt.show()


def get_name(file_path, changing):
    """
    returns the name of the file with number of ancillas/ transparencies used.
    :param changing:
    :param file_path: string - absolute path to given file
    :return: string of the format - "# ancillas/transparencies"
    """
    name = file_path.split("\\")[-1].split(",")[1]
    if changing == ANCILLAS:
        return name.split()[0]
    elif changing == TRANS:
        return name.split('.')[0].split()[0]  # removes .csv


if __name__ == '__main__':
    plot_err("GBphsmskImg.bmp", "change_transparencies_1", TRANS)
    # plot_err("GBphsmskImg.bmp", "change_transparencies_2", TRANS)


