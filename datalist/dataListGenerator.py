import os
import sys
import csv
from random import shuffle


def collect_all_h5_files(dir_name):
    '''
    collect all files in a base directory, recursively
    :param dir_name: base directory name
    :return: all h5 files in this directory, including the files in subdirectories.
    '''
    train_files = []
    for r, d, f in os.walk(dir_name):
        for file in f:
            if '.h5' in file:
                train_files.append(os.path.join(r, file))
    return train_files


def dump_csv(flist, outfname):
    with open(outfname, 'w') as train_out_f:
        train_out_f.write("Name\n")
        for tfname in flist:
            train_out_f.write(tfname + "\n")


def run(basedir, dataset_name):
    train_d = os.path.join(basedir, "train")
    val_d = os.path.join(basedir, "val")
    train_flist = collect_all_h5_files(train_d)
    val_flist = collect_all_h5_files(val_d)
    shuffle(train_flist)
    shuffle(val_flist)
    train_out_fname = "{}_hdf5_train.csv".format(dataset_name)
    val_out_fname = "{}_hdf5_val.csv".format(dataset_name)
    dump_csv(train_flist, train_out_fname)
    dump_csv(val_flist, val_out_fname)


# # here I'm generating the kitti datset. you can generate your own dataset using this script
# dataset_base_dir = "/mnt/sda1/Dataset/Kitti/hdf5/kitti"
# # generate train and val dataset path
# train_d = os.path.join(dataset_base_dir, "train")
# val_d = os.path.join(dataset_base_dir, "val")
# # collect all h5 files in the train and val direcotry
# train_flist = collect_all_h5_files(train_d)
# val_flist = collect_all_h5_files(val_d)
# # shuffle the file names.
# shuffle(train_flist)
# shuffle(val_flist)
# # save into the csv file
# train_out_fname = "kittidepth_hd5_train.csv"
# val_out_fname = "kittidepth_hd5_train.csv"



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('usage:\npython dataListGenerator.py your_h5_dataset_base_dir dataset_name')
    else:
        bdir = sys.argv[1]
        oname = sys.argv[2]
        run(bdir, oname)