#!/usr/bin/env bash
# -*- coding: utf-8 -*-

algorithm_dir=/opt/huawei/schedule-train/algorithm
code_path=$algorithm_dir/python/DL_algorithm/GraphEmbedding
date_train=$date_train
model_path=$OUTPUT_DIR
data_path=$data_dir
data2_dir=$data2_dir
split_str=$split_str
LOG_DIR=$LOG_DIR
embedding_name=$embedding_name
echo $algorithm_dir
echo $data_dir
echo $data2_dir
echo $OUTPUT_DIR
echo $date_train
echo $embedding_name
echo $split_str


if python -u $code_path/tensorboard.py ${date_train} --embedding_name=$embedding_name --split_str=$split_str --OUTPUT_DIR=$model_path --data_dir=$data_path --data2_dir=$data2_dir --LOG_DIR=$LOG_DIR;then echo "训练OK";else exit 1;fi
