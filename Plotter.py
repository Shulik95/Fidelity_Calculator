# ---------- imports ----------- #

import Fidelity_calculator as fc
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

# ----------- macros ----------- #
ANCILLAS = 1
TRANS = 2
RID = 3
DEFAULT_SUB_IMG_PATH = "Found sub-images"
DEFAULT_CSV_PATH = "csv data"
DEFAULT_ORIG_PATH = "target_dir"


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


def runner():
    """
    main runner code fore entire program assumes that the original image is
    in the working folder and is called OGimage.png. also assumes that if sub
    images are given as csv files they are in a directory called "csv data". else assumes
    the given images are in a directory called "png data".
    :return:
    """
    param_arr = ["Ancillas", "Transparencies", "Rydberg"]
    is_csv = input("Are the sub-images given as .csv?(y/n):")
    name = input("name of the original image (including the type):")
    var = int(input("What is the changing parameter?(1-ancillas, 2-transparencies, 3-radius"))
    symmetry = int(input("Is the symmetry horizontal or vertical? (1-horizontal, 2-vertical): "))
    if is_csv == "y":  # sub images exist in is_csv folder
        plot_err("OGimage.png", DEFAULT_CSV_PATH, var, 1)

    else:
        # TODO: complete section after helper functions are done
        err_arr = __create_sub_img_folder(DEFAULT_ORIG_PATH, name, symmetry, var)
        x = np.array([str(tup[1]) for tup in err_arr])
        num_of_sub_images = ['1', '2', '3']
        error = None
        for item in err_arr:
            col = []
            temp = 0
            for k in [0, 1, 2]:
                temp += item[0][k] if len(item[0]) >= k + 1 else -1
                col.append(temp / (k + 1))
            if error is None:
                error = np.array(col)
            else:
                error = np.column_stack((error, np.array(col)))
        fig, ax = plt.subplots()
        im, cbar = __heatmap(error, num_of_sub_images, x, ax=ax, cmap="PuOr", cbarlabel="SSIM Error")
        texts = annotate_heatmap(im, valfmt="{x:.2f}")

        fig.tight_layout() , plt.title(), plt.ylabel("# of sub-images"), plt.ylabel("# of ancillas")
        plt.savefig("SSIM_err_heatmap")
        ax.set_title("SSIM Error vs # of ancillas & # of sub-images")
        plt.show()


def __heatmap(data, row_labels, col_labels, ax=None,
              cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def __find_min_err(img_path, orig_img_path, symmetry):
    """
    find the sub-image with the minimal error, and stores it in a predefined
    directory for further use.
    :param img_path: path of the image to extract sub-images from, assumes image is png/jpeg/jpg
    :param orig_img_path: path of the original image to compare against.
    :return: avg SSIM score for given image.
    """

    # get array of found sub images
    img = fc.read_image(img_path)
    orig_img = fc.read_image(orig_img_path)
    cont_arr = fc.find_contours(fc.threshold_image(fc.filter_image(img), 180)[1])
    sub_shape_arr = fc.mark_contours(cont_arr, img, symmetry)
    if len(sub_shape_arr) == 0:  # no sub-images found, return -1
        return [-1]

    # calc avg error
    SSIM_score, idx = 0, 0
    # for obj in sub_shape_arr:
    # SSIM_score += fc.compare_img(orig_img, obj)[1]
    # name = "Sub-Shape" + str(idx) + ".png"
    # path = DEFAULT_SUB_IMG_PATH + str(idx) + "/" + name
    # # cv2.imwrite(path, obj) #TODO: fix subimage folders
    # idx += 1
    # SSIM_score = fc.compare_img(orig_img, obj)[1]

    SSIM_arr = []
    for item in sub_shape_arr:
        SSIM_arr.append(fc.compare_img(orig_img, item)[1])
    SSIM_arr.sort(reverse=True)
    return SSIM_arr  # return sorted sub array


def __create_sub_img_folder(target_dir, orig_img_path, symmetry, param):
    """
    iterates over all images in given target folder and creates sub-image folder
    for error comparison.
    :param target_dir - path to the directory containing the original images.
    :param orig_img_path - path to the original image
    :param symmetry - integer, 1 for horizontal symmetry, 2 for vertical.
    :return err_arr - array of arrays containing errors for sub images for each image
    """
    err_arr = []
    img_arr = sorted(os.listdir(target_dir), key=lambda x: int(x.split(",")[param].split()[0]))
    # handle directory

    # TODO: create a directory for sub-images of each of the images.
    for j in range(len(img_arr)):
        tmp_name = DEFAULT_SUB_IMG_PATH + str(j)
        if not os.path.isdir(tmp_name):  # no directory, create new one
            os.mkdir(tmp_name)
        else:  # directory exists, clear it.
            for filename in os.listdir(tmp_name):
                os.remove(os.path.join(tmp_name, filename))

    for file in img_arr:
        temp_path = os.path.abspath(os.path.join(file, os.pardir)) + "/" + target_dir + "/" + file
        err_arr.append((__find_min_err(temp_path, orig_img_path, symmetry), str(file).split(",")[param].split()[0]))
    return err_arr


if __name__ == '__main__':
    runner()
