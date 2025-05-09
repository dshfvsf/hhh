#!/bin/sh

#
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
#

#内部变量
algorithm_path=${ALGO_DIR:-"D:\AM_AppGallery_RecAlgorithmService"}
test_path="D:\AM_AppGallery_RecAlgorithmService\python\DL_algorithm\dev_test"
code_path=${algorithm_path}/DL_algorithm
if [ ! -e ${code_path} ]
then
    code_path=${algorithm_path}/python/DL_algorithm
fi
data_path=${data_dir:-"${test_path}\data_dir"}
target_path=${target_dir:-"${test_path}\target_dir"}
model_config_path=${target_path}/config
model_path=${OUTPUT_DIR:-"${test_path}\output_dir"}
log_path=${LOG_DIR:-"${test_path}\log_dir"}
dataset_log_path=$log_dir
config_dir=${CONFIG_DIR:-"${test_path}\config_dir"}
start_day="20250113"
end_day="20250115"
data_proc_flag=false

echo "-----------------running model train task----------------------"

gpu_num=$gpu_num
if [ -z "${gpu_num}" ]
then
    gpu_num=1
fi
#结束时间
day_end=${end_day}
if [ -z "${day_end}" ]
then
    #获取最近数据日期
    day_end=`ls ${data_path} | grep '^[0-9]\{8\}$' | awk 'BEGIN {max=0} {if($1>max) max=$1} END {print max}'`
    #最近数据天数阈值
    days_num=$days_nearest
    if [ -z "${days_num}" ]
    then
        days_num=5
    fi
    #判断最近数据是否满足条件
    day_at_least=`date -u -d "-${days_num} days +8 hours" +%Y%m%d`
    if [ ${day_end} -lt ${day_at_least} ]
    then
        echo "new data not arrived..."
        exit 1
    fi
fi

#开始时间
day_start=${start_day}
if [ -z "${day_start}" ]
then
    #训练数据窗口大小
    window_num=$days_train
    if [ -z "${window_num}" ]
    then
        window_num=30
    fi
    day_start=`date -u -d "${day_end} -${window_num} days" +%Y%m%d`
fi

#是否处理数据
flag_data_proc=$data_proc_flag
if [ -z "${flag_data_proc}" ]
then
    flag_data_proc=true
fi

#训练配置参数
orgi_config_dir=${config_dir}/train.config
config_path=${config_dir}/config_task.py
cp ${orgi_config_dir} ${config_path}

echo "--------------------print env variable-------------------------"
echo "start day: ${day_start}"
echo "end day: ${day_end}"
echo "algorithm_dir: ${algorithm_path}"
echo "config_path: ${config_path}"
echo "data_path: ${data_path}"
echo "target_path: ${target_path}"
echo "model_config_path: ${model_config_path}"
echo "model_path: ${model_path}"
echo "gpu num: ${gpu_num}"

# 1.data_process
if [ "${flag_data_proc}" = "true" ]
then
    echo "-------------------start processing data-----------------------"
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
fi

# 2.train model
#是否训练模型
if [ -z "${train_model}" ]
then
    train_model=true
fi

if [ "${train_model}" = "true" ]
then
    echo "-------------------start training model------------------------"
    if [ "$gpu_num" = "1" ]
    then
        echo "one gpu run"
        python -u ${code_path}/train/main.py ${day_end} \
        --config_task_file=${config_path} \
        --data_dir=${target_path} \
        --OUTPUT_DIR=${model_path} \
        --LOG_DIR=${log_path}
        if [ $? -eq 0 ]
        then
            echo "model training finished."
        else
            exit 1
        fi
    else
        echo "multi gpu run"
        mpirun --allow-run-as-root --tag-output -np $gpu_num -H localhost:$gpu_num -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib python -u ${code_path}/train/main.py ${day_end} \
        --config_task_file=${config_path} \
        --data_dir=${target_path} \
        --OUTPUT_DIR=${model_path} \
        --LOG_DIR=${log_path}

        echo $LOG_DIR
        log_file=$(ls $LOG_DIR|grep log)
        echo "log file $log_file"
        save_ok=$(grep -c "finish converting model to pb format" $LOG_DIR/$log_file)
        if [ $save_ok -eq 1 ]
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

    #4. copy log file to dataset
    if [ -e ${dataset_log_path} ]
    then
        echo $LOG_DIR
        log_file=$(ls $LOG_DIR|grep log)
        echo "log file $log_file"
        cp -f $LOG_DIR/$log_file $dataset_log_path/$log_file
        if [ $? -eq 0 ]
        then
            echo "copy log file finished."
        else
            echo "copy log file failed."
        fi
    fi
fi