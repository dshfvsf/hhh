#!/bin/sh

algorithm_path=$ALGO_DIR
code_path=${algorithm_path}/DL_algorithm
if [ ! -e ${code_path} ]
then
    code_path=${algorithm_path}/python/DL_algorithm
fi
data_path=$data_dir
target_path=$target_dir
model_config_path=${target_path}/config
model_path=$OUTPUT_DIR
log_path=$LOG_DIR

echo "-----------------running model train task----------------------"

day_end=$end_day
if [ -z "${day_end}" ]
then
    day_end=`ls ${data_path} | grep '^[0-9]\{8\}$' | awk 'BEGIN {max=0} {if($1>max) max=$1} END {print max}'`
    days_num=$days_nearest
    if [ -z "${days_num}" ]
    then
        days_num=5
    fi
    day_at_least=`date -u -d "-${days_num} days +8 hours" +%Y%m%d`
    echo "end day: ${day_end}"
    echo "day_at_least: ${day_at_least}"
    if [ ${day_end} -lt ${day_at_least} ]
    then
        echo "new data not arrived..."
        exit 1
    fi
fi

day_start=$start_day
if [ -z "${day_start}" ]
then
    window_num=$days_train
    if [ -z "${window_num}" ]
    then
        window_num=30
    fi
    day_start=`date -u -d "${day_end} -${window_num} days" +%Y%m%d`
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
echo "algorithm_dir: ${algorithm_path}"
echo "config_path: ${config_path}"
echo "data_path: ${data_path}"
echo "target_path: ${target_path}"
echo "model_config_path: ${model_config_path}"
echo "model_path: ${model_path}"

# 1.data_process
if [ "${flag_data_proc}" = "true" ]
then
    echo "-------------------start processing data-----------------------"
    echo "start day: ${day_start}"
    echo "end day: ${day_end}"
    python -u ${code_path}/data_process/process_Bi_data_common_feature.py ${day_start} ${day_end} \
    --day_end_as_testset \
    --incremental=false \
    --config_task_file=${config_path} \
    --data_dir=${data_path} \
    --target_dir=${target_path}

    if [ $? -eq 0 ]
    then 
        echo "finish processing data from ${day_start} to ${day_end}."
    else
        exit 1
    fi
    echo ${day_start} > ${target_path}/cur_dates.txt
    echo ${day_end} >> ${target_path}/cur_dates.txt
fi

# 2.train model
if [ "${flag_training}" = "true" ]
then
    echo "-------------------start training model------------------------"

    day_start=`head -n +1 ${target_path}/cur_dates.txt`
    day_end=`tail -n -1 ${target_path}/cur_dates.txt`
    echo "start day: ${day_start}"
    echo "end day: ${day_end}"

    python -u ${code_path}/train/main.py ${day_end} \
    --config_task_file=${config_path} \
    --data_dir=${target_path} \
    --llm_emb_dir=${llm_emb_dir} \
    --other_dir=${other_dir} \
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

echo "start day: ${day_start}"
echo "end day: ${day_end}"