#!/usr/bin/env bash
# -*- coding: utf-8 -*-

algorithm_dir=/opt/huawei/schedule-train/algorithm
code_path=$algorithm_dir/python/DL_algorithm/GraphEmbedding
date_train=$date_train
model_path=$OUTPUT_DIR
data_path=$data_dir
LOG_DIR=$LOG_DIR
model_train=$model_train
push_dir=$push_dir
embedding_name=$embedding_name
window=$window
min_count=$min_count
sg=$sg
echo $algorithm_dir
echo $data_dir
echo $target_dir
echo $OUTPUT_DIR
echo $date_train
echo $model_train
echo $embedding_name
echo $push_dir
echo $window
echo $min_count
echo $sg

if python -u $code_path/word2vec.py ${date_train} --model_train=$model_train --embedding_name=$embedding_name --window=$window  --min_count=$min_count  --sg=$sg --push_dir=$push_dir --OUTPUT_DIR=$model_path --data_dir=$data_path --LOG_DIR=$LOG_DIR;then echo "训练OK";else exit 1;fi
