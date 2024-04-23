"""
======================================================================
6.1.CORRELATION_COMPUTATION ---

Compute the correlation of different subfigures.

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2024, ZiLiang, all rights reserved.
    Created: 15 April 2024
======================================================================
"""


# ------------------------ Code --------------------------------------
## normal import 
import json
from typing import List,Tuple,Dict
import random
from pprint import pprint as ppp
from collections import OrderedDict

import numpy as np
from scipy.stats import spearmanr, pearsonr

def main():
    """
    XXX
    ------
    : 
    ------
    result:
    """
    # from collections import OrderedDict
    with open("./evaluate.6--input_ppl-ur.json",
              'r',encoding='utf8') as f:
        data=json.load(f,object_pairs_hook=OrderedDict)

    model_ls=list(data.keys())

    ngram_sim_ls=["5", "8", "11", "14"]
    fuzzy_sim_ls=["70", "80", "90", "100"]

    model_metric_srcorrelation_triple=[]
    model_metric_prcorrelation_triple=[]

    for model_name in model_ls:
        res_ls=data[model_name]
        ppl_ls=[x[0] for x in res_ls]
        for ngram in ngram_sim_ls:
            ngramres_ls=[x[1][ngram] for x in res_ls]

            # compute the correlation now.
            sr=spearmanr(ngramres_ls,ppl_ls)
            pr=pearsonr(ngramres_ls,ppl_ls)
            model_metric_srcorrelation_triple.append(
                [model_name,
                 ngram,
                 sr]
                )
            model_metric_prcorrelation_triple.append(
                [model_name,
                 ngram,
                 pr]
                )
        for roufuzzy in fuzzy_sim_ls:
            ngramres_ls=[x[2][roufuzzy] for x in res_ls]

            # compute the correlation now.
            pr=pearsonr(ngramres_ls,ppl_ls)
            sr=spearmanr(ngramres_ls,ppl_ls)
            model_metric_srcorrelation_triple.append(
                [model_name,
                 roufuzzy,
                 sr]
                )
            model_metric_prcorrelation_triple.append(
                [model_name,
                 roufuzzy,
                 pr]
                )
    print("==================")
    ppp(model_metric_srcorrelation_triple)
    print("==================")
    print("------------------")
    ppp(model_metric_prcorrelation_triple)
    print("------------------")

    return 0
 

## running entry
if __name__=="__main__":
    main()
    print("EVERYTHING DONE.")


