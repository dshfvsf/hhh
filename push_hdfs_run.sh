#!/usr/bin/env bash
# -*- coding: utf-8 -*-

algorithm_dir=/opt/huawei/schedule-train/algorithm
code_path=$algorithm_dir/python/DL_algorithm/common_util
model_path=$OUTPUT_DIR
data_path=$data_dir
LOG_DIR=$LOG_DIR
push_any=$push_any
model_name=$model_name
push_model_name=$push_model_name

push_dir=$push_dir/$(date "+%Y%m%d")

echo $algorithm_dir
echo $data_dir
echo $OUTPUT_DIR
echo $push_dir

# 判断上级目录是否存在
hadoop fs -test -e ${push_dir}
if [ $? -eq 1 ];then
    hadoop fs -mkdir -p ${push_dir}
    echo "mkdir done......"
	else
	hadoop fs -rm -r ${push_dir}
	echo "rmdir done......"
	hadoop fs -mkdir -p ${push_dir}
	echo "mkdir done......"
fi

if python -u $code_path/push_hdfs.py ${date_train}  --model_name=$model_name --push_model_name=$push_model_name --push_dir=$push_dir --OUTPUT_DIR=$model_path --data_dir=$data_path --push_any=$push_any --LOG_DIR=$LOG_DIR;then echo "push done";else exit 1;fi


