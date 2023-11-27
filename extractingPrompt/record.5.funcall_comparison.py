"""
======================================================================
RECORD.5.FUNCALL_COMPARISON ---

Curves of funcation calling.

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 26 November 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp

from matplotlib import pyplot as plt
import matplotlib
from sklearn import metrics
import sklearn
import sys
import numpy as np
from collections import OrderedDict
from metrics import ngram_recall_evaluate, fuzzy_match_recall

from transformers import AutoTokenizer


def plot_curves():
    prefix = "./func_callll/Llama-2-7b-chat-hf"
    funcpth = prefix+"#Func-res.json"
    normpth = prefix+"#Norm-res.json"

    model_types = {funcpth: "Json format Function Callings",
                   normpth: "Normal prompts",
                   }
    color_map = {funcpth: "red",
                 normpth: "blue", }
    marker_map = {
        funcpth: "o",
        normpth: "s",
    }
    line_map = {funcpth: "-", normpth: "-."}
    t = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf",
                                      trust_remote_code=True)

    n_ls = [12, 24, 36, 48]
    fuzzy_ls = [70, 80, 90, 100]

    font_size = 21

    j = 0
    fig, axs = plt.subplots(2, 4, figsize=(20, 9.3))
    for pth, m in model_types.items():
        with open(pth, 'r', encoding='utf8') as f:
            data = json.load(f, object_pairs_hook=OrderedDict)

        new_dict = {}
        for ap in data:
            for x_str in data[ap]:
                pair_ls = data[ap][x_str]

                for in_p, gen_p in pair_ls:
                    if in_p not in new_dict:
                        new_dict[in_p] = []
                    new_dict[in_p].append(gen_p)
        # then we get `new_dict` with inputPrompt-gen_prompts
        in_ps = list(new_dict.keys())
        lens = []
        for p in in_ps:
            lens.append(len(t(p, return_tensors="pt").input_ids[0]))

        for i_n, n in enumerate(n_ls):
            ylabel = f"{n}-gram UR"
            genss = [new_dict[p] for p in in_ps]
            res_ls = []
            for i_p, p in enumerate(in_ps):
                # print("-------------")
                # print(len(genss[i_p]))
                # print(len([p for _ in range(len(genss[i_p]))]))
                res_ls.append(ngram_recall_evaluate(genss[i_p],
                                                    [p for _ in range(
                                                        len(genss[i_p]))],
                                                    n=n
                                                    ))

            axs[0][i_n].set_xlabel("# of Tokens", fontsize=font_size)
            axs[0][i_n].set_ylabel(ylabel, fontsize=font_size-5)
            axs[0][i_n].tick_params(axis='y', labelsize=font_size-6,
                                    rotation=65,
                                    width=2, length=2,
                                    pad=0, direction="in",
                                    which="both")

            axs[0][i_n].scatter(lens, res_ls, label=m,
                                linewidth=1.5,
                                marker=marker_map[pth],
                                alpha=0.4,
                                color=color_map[pth],
                                )  # 绘制当前模型的曲线

        for i_n, n in enumerate(fuzzy_ls):
            ylabel = f"{n}% Fuzzy\nMatch UR"
            if n == 100:
                ylabel = r"$\mathbf{100\%}$"+" Fuzzy\nMatch UR"
            genss = [new_dict[p] for p in in_ps]
            res_ls = []
            for i_p, p in enumerate(in_ps):
                # print("-------------")
                # print(len(genss[i_p]))
                # print(len([p for _ in range(len(genss[i_p]))]))
                res_ls.append(fuzzy_match_recall(genss[i_p],
                                                 [p for _ in range(
                                                     len(genss[i_p]))],
                                                 ratio=n
                                                 ))

            axs[1][i_n].set_xlabel("# of Tokens", fontsize=font_size)
            axs[1][i_n].set_ylabel(ylabel, fontsize=font_size-5)
            axs[1][i_n].tick_params(axis='y', labelsize=font_size-6,
                                    rotation=65,
                                    width=2, length=2,
                                    pad=0, direction="in",
                                    which="both")

            axs[1][i_n].scatter(lens, res_ls, label=m,
                                linewidth=1.5,
                                marker=marker_map[pth],
                                alpha=0.4,
                                color=color_map[pth],
                                )  # 绘制当前模型的曲线

    fig.subplots_adjust(wspace=0.30, hspace=1.1)
    font1 = {
        'weight': 'normal',
        'size': font_size-1,
    }
    plt.legend(loc=(-4.18, 1.05),
               prop=font1, ncol=6, frameon=False,
               handletextpad=0., handlelength=1.2)  # 设置信息框
    plt.subplots_adjust(bottom=0.33, top=0.85)
    # plt.show()
    plt.savefig("./func_call_scatter.pdf",
                pad_inches=0.1)


def box_curves():
    prefix = "./func_callll/Llama-2-7b-chat-hf"
    funcpth = prefix+"#Func-res.json"
    normpth = prefix+"#Norm-res.json"

    model_types = {funcpth: "Json format Function Callings",
                   normpth: "Normal prompts",
                   }
    color_map = {funcpth: "red",
                 normpth: "blue", }
    marker_map = {
        funcpth: "o",
        normpth: "s",
    }
    # t = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf",
    #                                   trust_remote_code=True)

    n_ls = [12, 24, 36, 48]
    fuzzy_ls = [70, 80, 90, 100]

    font_size = 21

    j = 0
    fig, axs = plt.subplots(2, 4, figsize=(20, 9.3))
    for pth, m in model_types.items():
        with open(pth, 'r', encoding='utf8') as f:
            data = json.load(f, object_pairs_hook=OrderedDict)

        new_dict = {}
        for ap in data:
            for x_str in data[ap]:
                if x_str not in new_dict:
                    new_dict[x_str] = {}
                pair_ls = data[ap][x_str]
                ps, genps = zip(*pair_ls)
                n_gram_dict = {}
                fuzzy_dict = {}
                for n in n_ls:
                    n_gram_dict[n] = ngram_recall_evaluate(genps, ps, n=n)
                for r in fuzzy_ls:
                    fuzzy_dict[r] = fuzzy_match_recall(genps, ps, ratio=r)

                new_dict[x_str][ap] = {
                    "ngram": n_gram_dict,
                    "fuzzy": fuzzy_dict,
                }

        # now plot the box plot.
        interval_str_ls = list(new_dict.keys())
        interval_value_ls = [int(float(x)) for x in interval_str_ls]

        for i_n, n in enumerate(n_ls):
            ylabel = f"{n}-gram UR"
            yls = []
            for interval_str in interval_str_ls:
                ys = []
                for ap in new_dict[interval_str]:
                    ys.append(new_dict[interval_str][ap]["ngram"][n])
                yls.append(ys)

            axs[0][i_n].set_xlabel("# of Tokens", fontsize=font_size)
            axs[0][i_n].set_ylabel(ylabel, fontsize=font_size-5)
            axs[0][i_n].tick_params(axis='y', labelsize=font_size-6,
                                    rotation=65,
                                    width=2, length=2,
                                    pad=0, direction="in",
                                    which="both")

            cr = color_map[pth]
            kr = marker_map[pth]
            boxes = axs[0][i_n].boxplot(yls,
                                        positions=interval_value_ls,
                                        widths=15.5,
                                        boxprops={"color": cr,
                                                  "linewidth": 1.5,
                                                  # "gid":5.5,
                                                  },
                                        capprops={"color": cr,
                                                  "linewidth": 1.5,
                                                  },
                                        whiskerprops={"color": cr,
                                                      "linewidth": 1.5,
                                                      },
                                        flierprops={
                                            "markeredgecolor": cr,
                                            "marker": kr,
                                        },

                                        showmeans=True,
                                        meanline=True,
                                        showfliers=False,
                                        # patch_artist=True,
                                        )
        for i_n, n in enumerate(fuzzy_ls):
            ylabel = f"{n}% Fuzzy\nMatch UR"
            if n == 100:
                ylabel = r"$\mathbf{100\%}$"+" Fuzzy\nMatch UR"
            yls = []
            for interval_str in interval_str_ls:
                ys = []
                for ap in new_dict[interval_str]:
                    ys.append(new_dict[interval_str][ap]["fuzzy"][n])
                yls.append(ys)

            axs[1][i_n].set_xlabel("# of Tokens", fontsize=font_size)
            axs[1][i_n].set_ylabel(ylabel, fontsize=font_size-5)
            axs[1][i_n].tick_params(axis='y', labelsize=font_size-6,
                                    rotation=65,
                                    width=2, length=2,
                                    pad=0, direction="in",
                                    which="both")

            cr = color_map[pth]
            kr = marker_map[pth]
            boxes = axs[1][i_n].boxplot(yls,
                                        positions=interval_value_ls,
                                        widths=15.5,
                                        boxprops={"color": cr,
                                                  "linewidth": 1.5,
                                                  # "gid":5.5,
                                                  },
                                        capprops={"color": cr,
                                                  "linewidth": 1.5,
                                                  },
                                        whiskerprops={"color": cr,
                                                      "linewidth": 1.5,
                                                      },
                                        flierprops={
                                            "markeredgecolor": cr,
                                            "marker": kr,
                                        },

                                        showmeans=True,
                                        meanline=True,
                                        showfliers=False,
                                        # patch_artist=True,
                                        )

    fig.subplots_adjust(wspace=0.30, hspace=1.1)
    font1 = {
        'weight': 'normal',
        'size': font_size-1,
    }
    plt.legend(loc=(-4.18, 1.05),
               prop=font1, ncol=6, frameon=False,
               handletextpad=0., handlelength=1.2)  # 设置信息框
    plt.subplots_adjust(bottom=0.33, top=0.85)
    # plt.show()
    plt.savefig("./funcalling_boxes.pdf",
                pad_inches=0.1)


# running entry
if __name__ == "__main__":
    # main()
    # plot_curves()
    box_curves()
    print("EVERYTHING DONE.")
