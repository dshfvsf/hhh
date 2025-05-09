#!/usr/bin/env bash
# -*- coding: utf-8 -*-

algorithm_dir=/opt/huawei/schedule-train/algorithm
code_path=$algorithm_dir/DL_algorithm/GraphEmbedding
date_train=$date_train
model_path=$OUTPUT_DIR
data_path=$data_dir
LOG_DIR=$LOG_DIR
model_train=True
push_dir=$push_dir
file_name=$file_name
echo $algorithm_dir
echo $data_dir
echo $target_dir
echo $OUTPUT_DIR
echo $date_train
echo $model_train
echo $push_dir
echo $file_name

if python -u $code_path/node2vec.py ${date_train} --model_train=$model_train --file_name=$file_name --push_dir=$push_dir\
 --OUTPUT_DIR=$target_dir --data_dir=$data_path --LOG_DIR=$LOG_DIR;then echo "训练OK";else exit 1;fi
