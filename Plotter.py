# ---------- imports ----------- #
import Fidelity_calculator as fc
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as im
from skimage.color import rgba2rgb, rgb2gray

# ----------- macros ----------- #
ANCILLAS = 1
TRANS = 2
RID = 3
DEFAULT_SUB_IMG_PATH = "Found sub-images"
DEFAULT_CSV_PATH = "csv data"
DEFAULT_ORIG_PATH = "png data"


# ------------ code ------------ #

def __img_from_csv(path, parameter):
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
        ret = [(np.abs(np.genfromtxt(file, delimiter=',', dtype=None, encoding=None)), __get_name(file, parameter)) for
               file in
               only_files]  # converts csv to np array

    elif os.path.isfile(path):  # handle single file
        name, extension = os.path.splitext(path)
        if extension != ".csv":
            print("Given path doesn't contain a .csv file")
            return
        ret.append((np.abs(np.genfromtxt(path, delimiter=",", dtype=None, encoding=None)), __get_name(path, parameter)))
    else:
        print("Given path is not a Directory or a file")
        return
    return sorted(ret, key=lambda x: int(x[1]))


def plot_err(orig_img_path, path, param, is_csv, err_func="ALL"):
    """

    :param is_csv: integer - 0 if comparison images are images, positive integer if csv
    :param orig_img_path: string - path to original image
    :param path: string - path to directory of images for comparison
    :param err_func: string - indicating which error function to use, default is all.
    :param param: int - indicating if the changing parameter is acillas or transparencies
    :return:
    """
    to_compare_lst, err_mat = None, None
    orig_img = fc.read_image(orig_img_path)
    to_compare_lst = __img_from_csv(path, param) if is_csv else __get_img_arr(path, param)

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
    temp = ''
    for i in range(err_mat.shape[1]):
        norm = max(err_mat.T[i])
        plt.scatter(x_arr, err_mat.T[i] / norm)
        plt.plot(x_arr, err_mat.T[i], label=titles[i])
        plt.xticks(x_arr)
        if param == ANCILLAS:
            plt.xlabel("# of ancillas")
            temp = "Ancillas"
        else:
            plt.xlabel("# of transparencies")
            temp = "Transparencies"
        # plt.ylabel("error")
        # plt.title(titles[i] + " vs. # of " + temp)
    # plt.legend(handles=titles)
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.title("Error vs. " + temp)
    plt.savefig("Error vs. " + temp + ".jpeg")
    plt.show()


def __get_name(file_path, changing):
    """
    returns the name of the file with number of ancillas/ transparencies used.
    :param changing:
    :param file_path: string - absolute path to given file
    :return: string of the format - "# ancillas/transparencies"
    """
    name = file_path.split("\\")[-1]
    if changing == ANCILLAS:
        return name.split(",")[1].split()[0]
    elif changing == TRANS:
        return name.split(',')[2].split()[0]  # removes .csv
    elif changing == RID:
        return name.split(',')[3].split()[0]


def __get_img_arr(path, parameter):
    """
    returns an array of images as np.array
    :param path: path of folder containing images.
    :param parameter: the changing variable in the program.
    :return: python list of tuples in the format (image, # of ancillas/transparencies) where image is np.array.
    """
    ret = []
    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith('.png') or filename.endswith('.jpg'):  # check if image
                ret.append((fc.read_image(filename), __get_name(filename, parameter)))
    else:
        print("Given path is not a Directory or a file")
        return
    return ret


# TODO: create general runner for different types of images
def runner():
    """
    main runner code fore entire program assumes that the original image is
    in the working folder and is called OGimage.png. also assumes that if sub
    images are given as csv files they are in a directory called "csv data". else assumes
    the given images are in a directory called "png data".
    :return:
    """
    is_csv = input("Are the sub-images given as .csv?(y/n):")
    var = int(input("What is the changing parameter?(1-ancillas, 2-transparencies, 3-radius"))
    symmetry = input("Is the symmetry horizontal or vertical? (1-horizontal, 2-vertical): ")
    if is_csv == "y":  # sub images exist in is_csv folder
        plot_err("OGimage.png", DEFAULT_CSV_PATH, var, 1)
    else:
        # TODO: complete section after helper functions are done
        # --- #
        plot_err("OGimage.png", DEFAULT_SUB_IMG_PATH, var, 0)


def __find_min_err(img_path, orig_img_path, symmetry):
    """
    find the sub-image with the minimal error, and stores it in a predefined
    directory for further use.
    :param img_path: path of the image to extract sub-images from, assumes image is png/jpeg/jpg
    :param orig_img_path: path of the original image to compare against.
    """

    # find min error sub-image
    max_score, max_img = float('-inf'), None
    img = fc.read_image(img_path)
    orig_img = fc.read_image(orig_img_path)
    cont_arr = fc.find_contours(fc.threshold_image(fc.filter_image(img), 200)[1])
    sub_shape_arr = fc.mark_contours(cont_arr, img, symmetry)
    for sub_img in sub_shape_arr:
        ssim_score = fc.compare_img(orig_img, sub_img)[1]
        if ssim_score > max_score:
            max_img, max_score = sub_img, ssim_score
    if max_img is None:
        return

    # save found image in designated folder
    temp_path = DEFAULT_SUB_IMG_PATH + "/" + img_path.split("/")[2]
    im.imsave(temp_path, max_img)


def __create_sub_img_folder(target_dir, orig_img_path, symmetry):
    """
    iterates over all images in given target folder and creates sub-image folder
    for error comparison.
    :return:
    """
    # handle directory
    if not os.path.isdir(DEFAULT_SUB_IMG_PATH):  # no directory, create new one
        os.mkdir(DEFAULT_SUB_IMG_PATH)
    else:  # directory exists, clear it.
        for filename in os.listdir(DEFAULT_SUB_IMG_PATH):
            os.remove(os.path.join(DEFAULT_SUB_IMG_PATH, filename))

    for file in os.listdir(target_dir):
        temp_path = os.path.abspath(os.path.join(file, os.pardir)) + "/" + target_dir + "/" + file
        __find_min_err(temp_path, orig_img_path, symmetry)


if __name__ == '__main__':
    __create_sub_img_folder("target_dir", "SCatW.bmp", 1)
