#!/bin/sh

algorithm_path=$ALGO_DIR
code_path=${algorithm_path}/DL_algorithm
if [ ! -e ${code_path} ]
then
    code_path=${algorithm_path}/python/DL_algorithm
fi
data_path=$data_dir_2
data_path_1=$data_dir
data_path_2=$data_dir_2
target_path=$target_dir_2
model_config_path=${target_path}/config
model_path=$OUTPUT_DIR
log_path=$LOG_DIR

target_path_2=$target_dir
target_path_3=$target_dir_2
target_path_4=$target_dir_4
target_path_5=$target_dir_5
echo "target_path_2: ${target_path_2}"
echo "target_path_3: ${target_path_3}"
echo "target_path_4: ${target_path_4}"
echo "target_path_5: ${target_path_5}"
echo "-----------------running model train task----------------------"


day_end_1=$end_day_1
if [ -z "${day_end_1}" ]
then
    day_end_1=`ls ${data_path_1} | grep '^[0-9]\{8\}$' | awk 'BEGIN {max=0} {if($1>max) max=$1} END {print max}'`
    days_num=$days_nearest
    if [ -z "${days_num}" ]
    then
        days_num=5
    fi
    day_at_least=`date -u -d "-${days_num} days +8 hours" +%Y%m%d`
    echo "end day: ${day_end_1}"
    echo "day_at_least: ${day_at_least}"
    if [ ${day_end_1} -lt ${day_at_least} ]
    then
        echo "new data not arrived..."
        exit 1
    fi
fi

day_start_1=$start_day_1
if [ -z "${day_start_1}" ]
then
    window_num=$days_train
    if [ -z "${window_num}" ]
    then
        window_num=30
    fi
    day_start_1=`date -u -d "${day_end_1} -${window_num} days" +%Y%m%d`
fi

day_end_2=$end_day_2
if [ -z "${day_end_2}" ]
then
    day_end_2=`ls ${data_path_2} | grep '^[0-9]\{8\}$' | awk 'BEGIN {max=0} {if($1>max) max=$1} END {print max}'`
    days_num=$days_nearest
    if [ -z "${days_num}" ]
    then
        days_num=5
    fi
    day_at_least=`date -u -d "-${days_num} days +8 hours" +%Y%m%d`
    echo "end day: ${day_end_2}"
    echo "day_at_least: ${day_at_least}"
    if [ ${day_end_2} -lt ${day_at_least} ]
    then
        echo "new data not arrived..."
        exit 1
    fi
fi

day_start_2=$start_day_2
if [ -z "${day_start_2}" ]
then
    window_num=$days_train
    if [ -z "${window_num}" ]
    then
        window_num=30
    fi
    day_start_2=`date -u -d "${day_end_2} -${window_num} days" +%Y%m%d`
fi

flag_data_proc=$data_proc_flag
if [ -z "${flag_data_proc}" ]
then
    flag_data_proc=true
fi

flag_training=$training_flag
if [ -z "${flag_training}" ]
then
    flag_training=true
fi

orgi_config_dir=${algorithm_path}/train.config
config_path=${code_path}/data_process/config_task.py
cp ${orgi_config_dir} ${config_path}

echo "--------------------print env variable-------------------------"
echo "start day 1: ${day_start_1}"
echo "end day 1: ${day_end_1}"
echo "start day 2: ${day_start_2}"
echo "end day 2: ${day_end_2}"
echo "algorithm_dir: ${algorithm_path}"
echo "config_path: ${config_path}"
echo "data_path: ${data_path}"
echo "target_path: ${target_path}"
echo "model_config_path: ${model_config_path}"
echo "model_path: ${model_path}"
echo "data_path_1: ${data_path_1}"
echo "target_path_1: ${target_path_2}"
echo "data_path_2: ${data_path_2}"
echo "target_path_2: ${target_path_3}"
# 1.data_process
if [ "${flag_data_proc}" = "true" ]
then
    echo "-------------------start processing data 1-----------------------"
    python -u ${code_path}/data_process/process_Bi_data_common_feature.py ${day_start_1} ${day_end_1} \
    --day_end_as_testset \
    --incremental=false \
    --config_task_file=${config_path} \
    --data_dir=${data_path_1} \
    --target_dir=${target_path_2}

    if [ $? -eq 0 ]
    then 
        echo "finish processing data 1 from ${day_start_1} to ${day_end_1}."
    else
        exit 1
    fi
    echo "-------------------start processing data 2-----------------------"
    python -u ${code_path}/data_process/process_Bi_data_common_feature.py ${day_start_2} ${day_end_2} \
    --day_end_as_testset \
    --incremental=false \
    --config_task_file=${config_path} \
    --data_dir=${data_path_2} \
    --target_dir=${target_path_3} \
    --ori_dir=${data_path_1} \
    --ori_day_end=${day_end_1} \
    --map_index

    if [ $? -eq 0 ]
    then 
        echo "finish processing data 2 from ${day_start_2} to ${day_end_2}."
    else
        exit 1
    fi
    
    echo "-------------------start merge information-----------------------"
    python -u ${code_path}/data_process/process_multi_dataset.py \
    --day_end_1=${day_end_1} \
    --day_end_2=${day_end_2} \
    --config_task_file=${config_path} \
    --target_dir_1=${target_path_2} \
    --target_dir_2=${target_path_3} \
    --data_dir_1=${data_path_1} \
    --data_dir_2=${data_path_2} 
    
    if [ $? -eq 0 ]
    then 
        echo "finish merge info."
    else
        exit 1
    fi
    
fi

# 2.train model
if [ "${flag_training}" = "true" ]
then
    echo "-------------------start training model------------------------"
    python -u ${code_path}/train/main.py ${day_end_2} \
    --day_end_1=${day_end_1} \
    --day_end_2=${day_end_2} \
    --config_task_file=${config_path} \
    --data_dir=${target_path_2} \
    --data_other_dir_2=${target_path_2} \
    --data_other_dir_3=${target_path_3} \
    --data_other_dir_4=${target_path_4} \
    --data_other_dir_5=${target_path_5} \
    --OUTPUT_DIR=${model_path} \
    --LOG_DIR=${log_path}

    if [ $? -eq 0 ]
    then
        echo "model training finished."
    else 
        exit 1
    fi
fi

#3.config file
echo "---------------------copy config file--------------------------"
if [ -e ${model_config_path} ]
then
    cp ${model_config_path}/* ${model_path}
    
    if [ $? -eq 0 ]
    then
        echo "copy config file finished."
    else 
        echo "copy config file failed."
    fi
fi

echo "----------------model train task finished----------------------"

