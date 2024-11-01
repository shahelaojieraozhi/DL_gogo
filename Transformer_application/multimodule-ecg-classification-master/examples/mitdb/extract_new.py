# -*- coding: utf-8 -*-
"""
@Project ：DL_gogo 
@Time    : 2023/10/22 8:45
@Author  : Rao Zhi
@File    : extract_new.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
from __future__ import print_function

import pickle
import glob
import numpy as np
import os
import subprocess
import wfdb
from matplotlib import pyplot as plt


def extract_wave(idx):
    """
    Reads .dat file and returns in numpy array. Assumes 2 channels.  The
    returned array is n x 3 where n is the number of samples. The first column
    is the sample number and the second two are the first and second channel
    respectively.
    """
    record = wfdb.rdrecord("../../data/mit-bih-arrhythmia-database-1.0.0/" + str(idx),  # 文件所在路径
                           # sampfrom=0,  # 读取100这个记录的起点，从第0个点开始读
                           # sampto=1000,  # 读取记录的终点，到1000个点结束
                           physical=False,  # 若为True则读取原始信号p_signal，如果为False则读取数字信号d_signal，默认为False
                           channels=[0, 1])  # 读取那个通道，也可以用channel_names指定某个通道;如channel_names=['MLII']

    # 转为数字信号
    signal = np.array(record.d_signal).astype(np.int32)
    signal_index = np.arange(len(signal)).reshape(-1, 1)
    return np.concatenate([signal_index, signal], axis=1)


def extract_annotation(idx):
    """
    The annotation file column names are:
        Time, Sample #, Type, Sub, Chan, Num, Aux
    The Aux is optional, it could be left empty. Type is the beat type and Aux
    is the transition label.
    """
    signal_ann = wfdb.rdann("../../data/mit-bih-arrhythmia-database-1.0.0/" + str(idx), "atr")

    # # 并打印出改心拍标注的类型
    # print(signal_ann.symbol)
    labels = signal_ann.symbol

    return labels


def extract(idx):
    """
    Extracts data and annotations from .dat and .atr files.
    Returns a numpy array for the data and a list of tuples for the labels.
    """
    data = extract_wave(idx)
    labels = extract_annotation(idx)
    return data, labels


def save(example, idx):
    """
    Saves data with numpy.save (load with numpy.load) and pickles labels. The
    files are saved in the same place as the raw data.
    """
    np.save(os.path.join(save_path, idx), example[0])
    with open(os.path.join(save_path, "{}.pkl".format(idx)), 'wb') as fid:
        pickle.dump(example[1], fid)


if __name__ == "__main__":
    WFDB = "/deep/group/med/tools/wfdb-10.5.24/build/bin/"
    DATA = "../../data/mit-bih-arrhythmia-database-1.0.0"
    save_path = "../../data/extracted_mit_bih/"
    files = glob.glob(os.path.join(DATA, "*.dat"))
    idxs = [os.path.basename(f).split(".")[0] for f in files]
    for idx in idxs:
        example = extract(idx)
        save(example, idx)
        print("Example {}".format(idx))


