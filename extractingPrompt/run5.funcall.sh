#!/bin/bash
######################################################################
#RUN5.FUNCALL --- 

# Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
# Copyright © 2023, ZiLiang, all rights reserved.
# Created: 22 November 2023
######################################################################

######################### Commentary ##################################
##  
######################################################################

export python=/home/liangzi/anaconda3/envs/attgpt4/bin/python3

export ProjectPath="/home/liangzi/code/attackFineTunedModels/extractingPrompt/"

cd ${ProjectPath}

nohup ${python} 5.funcall_comparison.py > 1122_funcall_res.log &

echo "Everything Done."


echo "RUNNING run5.funcall.sh DONE."
# run5.funcall.sh ends here
