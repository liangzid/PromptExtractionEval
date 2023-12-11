"""
======================================================================
VISUALIZE_ATTENTION_ALPHABETAGAMMA ---

After running `attention_visualize.py`

    Author: Zi Liang <frost.liang@polyu.edu.hk>
    Copyright © 2023, ZiLiang, all rights reserved.
    Created: 10 December 2023
======================================================================
"""


# ------------------------ Code --------------------------------------

# normal import
import json
from typing import List, Tuple, Dict
import random
from pprint import pprint as ppp

import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict


def plot_heatmap(data_neg_pth="./attention_viz/Sampled__NEGIVE_0_img---metricsRes_layer24_head32.json",
                 data_pos_pth="./attention_viz/Sampled__POSITIVE_0_img---metricsRes_layer24_head32.json"):

    with open(data_pos_pth, 'r', encoding='utf8') as f:
        datap = json.load(f, object_pairs_hook=OrderedDict)
    with open(data_neg_pth, 'r', encoding='utf8') as f:
        datan = json.load(f, object_pairs_hook=OrderedDict)

    Nl = 24
    Nh = 32
    p_mat_ap = np.zeros((Nl, Nh))
    p_mat_an = np.zeros((Nl, Nh))
    p_mat_bp = np.zeros((Nl, Nh))
    p_mat_gp = np.zeros((Nl, Nh))
    p_mat_gn = np.zeros((Nl, Nh))

    n_mat_ap = np.zeros((Nl, Nh))
    n_mat_an = np.zeros((Nl, Nh))
    n_mat_bp = np.zeros((Nl, Nh))
    n_mat_gp = np.zeros((Nl, Nh))
    n_mat_gn = np.zeros((Nl, Nh))

    for nl in range(Nl):
        for nh in range(Nh):
            data = datap[str(nl)][str(nh)]
            p_mat_ap[nl][nh] = data["alpha_p"]
            p_mat_an[nl][nh] = data["alpha_n"]
            p_mat_bp[nl][nh] = data["beta_p"]
            p_mat_gp[nl][nh] = data["gammar_p"]
            p_mat_gn[nl][nh] = data["gammar_n"]

            data = datan[str(nl)][str(nh)]
            n_mat_ap[nl][nh] = data["alpha_p"]
            n_mat_an[nl][nh] = data["alpha_n"]
            n_mat_bp[nl][nh] = data["beta_p"]
            n_mat_gp[nl][nh] = data["gammar_p"]
            n_mat_gn[nl][nh] = data["gammar_n"]

    # finally draw figures

    fig, axs = plt.subplots(2, 5, figsize=(20, 6.8))

    res = axs[0, 0].imshow(p_mat_ap,
                           cmap=plt.cm.Blues,
                           )
    res = axs[0, 1].imshow(p_mat_an,
                           cmap=plt.cm.Blues,
                           )
    res = axs[0, 2].imshow(p_mat_bp,
                           cmap=plt.cm.Blues,
                           )
    res = axs[0, 3].imshow(p_mat_gp,
                           cmap=plt.cm.Blues,
                           )
    res = axs[0, 4].imshow(p_mat_gn,
                           cmap=plt.cm.Blues,
                           )

    res = axs[1, 0].imshow(n_mat_ap,
                           cmap=plt.cm.Blues,
                           )
    res = axs[1, 1].imshow(n_mat_an,
                           cmap=plt.cm.Blues,
                           )
    res = axs[1, 2].imshow(n_mat_bp,
                           cmap=plt.cm.Blues,
                           )
    res = axs[1, 3].imshow(n_mat_gp,
                           cmap=plt.cm.Blues,
                           )
    res = axs[1, 4].imshow(n_mat_gn,
                           cmap=plt.cm.Blues,
                           )

    lw = 0.8
    # axs.axhline(y=end_p, color="red", xmin=0, xmax=end_p,
    #             linewidth=lw)
    # axs.axhline(y=bgn_genp, color="red", xmin=0, xmax=bgn_genp,
    #             linewidth=lw)
    # axs.axhline(y=end_genp, color="red", xmin=0, xmax=end_genp,
    #             linewidth=lw)

    # axs.axvline(x=end_p, color="red", ymin=newlen-end_p, ymax=newlen,
    #             linewidth=lw)
    # axs.axvline(x=bgn_genp, color="red", ymin=newlen-bgn_genp, ymax=newlen,
    #             linewidth=lw)
    # axs.axvline(x=end_genp, color="red", ymin=newlen-end_genp, ymax=newlen,
    #             linewidth=lw)

    fs = 13
    for a in range(2):
        for b in range(5):
            axs[a, b].set_xlabel('Head Number',fontsize=fs)
            axs[a, b].set_ylabel('Layer Number',fontsize=fs)
            # plt.colorbar(res, ax=axs[a, b])
            if a == 0:
                p1 = 'Positive Samples'
            else:
                p1 = 'Negtive Samples'
            p2_map = {
                0: r"$\alpha_{pre}$",
                1: r"$\alpha_{cur}$",
                2: r"$\beta_{pre}$",
                3: r"$\gamma_{pre}$",
                4: r"$\gamma_{cur}$",
            }
            axs[a, b].title.set_text(p2_map[b]+" of " + p1,
                                     )

    plt.savefig(f"./attention_viz/indicators_comparison_singleSample.pdf",
                pad_inches=0.1)
    print("save done.")


# running entry
if __name__ == "__main__":
    plot_heatmap()
    print("EVERYTHING DONE.")
