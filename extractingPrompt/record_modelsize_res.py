"""
======================================================================
RECORD_MODELSIZE_RES ---

As the filename stated.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 20 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
from matplotlib import pyplot as plt
import matplotlib
from sklearn import metrics
import sklearn
import sys
import numpy as np
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp

from data_2 import Prompt_res_dict
from collections import OrderedDict

model_parameter_map = OrderedDict({
    "70m-deduped": 70,
    "160m-deduped": 160,
    "410m-deduped": 410,
    "1b-deduped": 1*1000,
    "1.4b-deduped": 1.4*1000,
    "2.8b-deduped": 2.8*1000,
    "6.9b-deduped": 6.9*1000,
    "12b-deduped": 12*1000,

})
model_parameter_map1 = OrderedDict({
    "125m": 125,
    "350m": 350,
    "1.3b": 1.3*1000,
    "2.7b": 2.7*1000,
    "6.7b": 6.7*1000,
    "13b": 13*1000,
})

model_xname_map = OrderedDict({
    "70m-deduped": "70M",
    "160m-deduped": "160M",
    "410m-deduped": "410M",
    "1b-deduped": "1B",
    "1.4b-deduped": "1.4B",
    "2.8b-deduped": "2.8B",
    "6.9b-deduped": "6.9B",
    "12b-deduped": "12B",

})

model_xname_map1 = OrderedDict({
    "125m": "125M",
    "350m": "350M",
    "1.3b": "1.3B",
    "2.7b": "2.7B",
    "6.7b": "6.7B",
    "13b": "13B",
})


marker = ['o', 's', 'o', 's',]  # 曲线标记
marker = {
    "E": "o",
    "I": "s",
}
model_color_dict = {
    # "E": "#FF0202",
    # "I": "#008000",
    "E": "red",
    "I": "green",
}
model_color_dict2=model_color_dict
# model_color_dict2 = {
#     # "E": "#FF0202",
#     # "I": "#008000",
#     "E": (252/255, 224/255, 225/255),
#     "I": (194/255, 232/255, 247/255),
# }

model_line_style = {
    "E": "-",
    "I": "-.",
}

font_size = 21


# n-gram list
fig_1to4_ls = [3, 6, 9, 12]

# ratio next
fig_5to8_ls = [70, 80, 90, 100]


def plot_figures():
    with open("./opt_varying_modelsize/scores.json",
              'r', encoding='utf8') as f:
        data1 = json.load(f, object_pairs_hook=OrderedDict)
    Prompt_res_dict1 = data1

    with open("./pythia_p_model_res/scores.json",
              'r', encoding='utf8') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    Prompt_res_dict = data

    j = 0
    fig, axs = plt.subplots(2, 4, figsize=(20, 9.3))
    # the first 4 images.
    for i_n, n in enumerate(fig_1to4_ls):
        axs[0][i_n].set_xscale("log")
        ylabel = f"{n}-gram UR"
        ngram_dict = {}
        for m in Prompt_res_dict.keys():
            ngram_dict[m] = {"E": {},
                             "I": {},
                             }
            for mode in Prompt_res_dict[m]:
                y = []
                for ap in Prompt_res_dict[m][mode]:
                    y.append(Prompt_res_dict[m][mode]
                             [ap]["ngram"][str(n)])
                ngram_dict[m][mode]["mean"] = sum(y)/len(y)
                ngram_dict[m][mode]["max"] = max(y)
                ngram_dict[m][mode]["min"] = min(y)
        print("ngram dict: ", ngram_dict)
        ngram_dict1 = {}
        for m in Prompt_res_dict1.keys():
            ngram_dict1[m] = {"E": {},
                              "I": {},
                              }
            for mode in Prompt_res_dict1[m]:
                y = []
                for ap in Prompt_res_dict1[m][mode]:
                    y.append(Prompt_res_dict1[m][mode]
                             [ap]["ngram"][str(n)])
                ngram_dict1[m][mode]["mean"] = sum(y)/len(y)
                ngram_dict1[m][mode]["max"] = max(y)
                ngram_dict1[m][mode]["min"] = min(y)
        print("ngram dict OPT: ", ngram_dict1)

        # ngram_dict = ngram_dict1
        # model_xname_map = model_xname_map1
        # model_parameter_map = model_parameter_map1

        mode = "E"
        for mode in ["E", "I"]:
            xs = list(model_xname_map.values())
            xvls = list(model_parameter_map.values())
            yls = [ngram_dict[mx][mode]["mean"] for mx in ngram_dict]
            ymin = [ngram_dict[mx][mode]["min"] for mx in ngram_dict]
            ymax = [ngram_dict[mx][mode]["max"] for mx in ngram_dict]
            print("xs and yls: ", xs, yls)
            axs[0][i_n].plot(xvls,
                             yls,
                             label="Explicit Attacking",
                             linewidth=1.5,
                             marker=marker[mode],
                             markevery=1, markersize=15,
                             markeredgewidth=1.5,
                             markerfacecolor='none',
                             alpha=1.,
                             linestyle=model_line_style[mode],
                             color=model_color_dict[mode]
                             )  # 绘制当前模型的曲线
            # 填充上下界区域内，设置边界、填充部分颜色，以及透明度
            axs[0][i_n].fill_between(xvls, ymin, ymax,
                                     alpha=0.2,
                                     # alpha=1.0,
                                     color=model_color_dict2[mode])  # 透明度
        axs[0][i_n].set_xlabel("Model Parameters", fontsize=font_size)
        axs[0][i_n].set_ylabel(ylabel, fontsize=font_size-5)
        axs[0][i_n].set_xticks(xvls, xs,
                               rotation=48, size=font_size-4)
        axs[0][i_n].tick_params(axis='y', labelsize=font_size-6,
                                rotation=65,
                                width=2, length=2,
                                pad=0, direction="in",
                                which="both")
    for i_n, n in enumerate(fig_5to8_ls):
        axs[1][i_n].set_xscale("log")
        ylabel = f"{n}% Fuzzy\nMatch UR"
        if n == 100:
            ylabel = r"$\mathbf{100\%}$"+" Fuzzy\nMatch UR"
        ngram_dict = {}
        for m in Prompt_res_dict.keys():
            ngram_dict[m] = {"E": {},
                             "I": {},
                             }
            for mode in Prompt_res_dict[m]:
                y = []
                axs[0][i_n].set_xscale("log")
                for ap in Prompt_res_dict[m][mode]:
                    y.append(Prompt_res_dict[m][mode]
                             [ap]["fuzzy"][str(n)])
                ngram_dict[m][mode]["mean"] = sum(y)/len(y)
                ngram_dict[m][mode]["max"] = max(y)
                ngram_dict[m][mode]["min"] = min(y)

        print("ngram dict: ", ngram_dict)
        mode = "E"
        for mode in ["E", "I"]:
            xs = list(model_xname_map.values())
            xvls = list(model_parameter_map.values())
            yls = [ngram_dict[mx][mode]["mean"] for mx in ngram_dict]
            ymin = [ngram_dict[mx][mode]["min"] for mx in ngram_dict]
            ymax = [ngram_dict[mx][mode]["max"] for mx in ngram_dict]
            print("xs and yls: ", xs, yls)
            axs[1][i_n].plot(xvls,
                             yls,
                             label="Explicit Attacking",
                             linewidth=1.5,
                             marker=marker[mode],
                             markevery=1, markersize=15,
                             markeredgewidth=1.5,
                             markerfacecolor='none',
                             alpha=1.,
                             linestyle=model_line_style[mode],
                             color=model_color_dict[mode]
                             )  # 绘制当前模型的曲线
            # 填充上下界区域内，设置边界、填充部分颜色，以及透明度
            axs[1][i_n].fill_between(xvls, ymin, ymax,
                                     alpha=0.2,
                                     color=model_color_dict2[mode],
                                     )  # 透明度
        axs[1][i_n].set_xlabel("Model Parameters", fontsize=font_size)
        axs[1][i_n].set_ylabel(ylabel, fontsize=font_size-5)
        axs[1][i_n].set_xticks(xvls, xs,
                               rotation=48, size=font_size-4)
        axs[0][i_n].set_xticks(xvls, xs,
                               rotation=48, size=font_size-4)
        axs[1][i_n].tick_params(axis='y', labelsize=font_size-6,
                                rotation=65,
                                width=2, length=2,
                                pad=0, direction="in",
                                which="both")

    # plt.legend(loc=(3.4, 5.8), prop=font1, ncol=6)  # 设置信息框
    # plt.legend(loc=(20, 1.5), prop=font1, ncol=6)  # 设置信息框
    font1 = {
        'weight': 'normal',
        'size': font_size-1,
    }
    plt.legend(loc=(-2.41, 2.60),
               prop=font1, ncol=6, frameon=False,
               handletextpad=0., handlelength=1.2)  # 设置信息框
    fig.subplots_adjust(wspace=0.26, hspace=0.6)
    plt.subplots_adjust(bottom=0.33, top=0.85)
    # plt.show()
    plt.savefig("./vary_params.pdf",
                pad_inches=0.1)


def main():
    plot_figures()


# running entry
if __name__ == "__main__":
    main()
    print("EVERYTHING DONE.")
