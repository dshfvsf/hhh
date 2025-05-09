#!/bin/bash
# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
# 分布式训练入口
# version: 1.0.0
# change log:
# ***********************************************************************

USE_SFPS=1
train_workdir=$(cd $(dirname $0); pwd)
echo train_workdir=${train_workdir}
echo workers=${workers}
echo args=$@

# echo ${sfps_param}

if [ $USE_SFPS -eq 1 ]
then
    echo "begin install sfps"

    echo "lib_ps is $lib_ps"
    cd $lib_ps/capnproto-c++-0.10.2
    chmod -R 777 $lib_ps/capnproto-c++-0.10.2
    ./configure
    make -j6 check
    make install

    cd /opt/huawei/sfps

    # rm -rf /opt/huawei/sfps-old && mkdir -p /opt/huawei/sfps-old && mv /opt/huawei/sfps/* /opt/huawei/sfps-old/
    # rm -rf /opt/huawei/sfps && mkdir -p /opt/huawei/sfps
    mkdir -p /opt/huawei/sfps
    cp -r $lib_ps/SFPS-0.8.5.1/* /opt/huawei/sfps/
    cd /opt/huawei/sfps && rm -rf /opt/huawei/sfps/SFPS/tools/*.pb.h && protoc -I=/opt/huawei/sfps/SFPS/tools/ --cpp_out=/opt/huawei/sfps/SFPS/tools/ /opt/huawei/sfps/SFPS/tools/KVEmbedding.proto

    algo_dir=/opt/huawei/sfps

    cd  $algo_dir/3rdparty/ps-lite/deps/lib/
    ln -s libzmq.so.5.0.0 libzmq.so.5
    ln -s libzmq.so.5 libzmq.so

    cd ${algo_dir}

    export pybind11_DIR=/usr/local/lib/python3.7/site-packages/pybind11/share/cmake/pybind11/
    export ZMQ_DIR=$algo_dir/3rdparty/ps-lite/deps/
    export RDMA=1
    export LD_LIBRARY_PATH=/opt/huawei/sfps/3rdparty/ps-lite/deps/lib:$LD_LIBRARY_PATH
    export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
    export USE_CAPNP=0
    ALL2ALL=0 pip install . -v
    df -lh

    echo "end install sfps"
fi


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
echo "start day: ${day_start}"
echo "end day: ${day_end}"
echo "algorithm_dir: ${algorithm_path}"
echo "config_path: ${config_path}"
echo "data_path: ${data_path}"
echo "target_path: ${target_path}"
echo "model_config_path: ${model_config_path}"
echo "model_path: ${model_path}"
echo "flag_data_proc: ${flag_data_proc}"
echo "flag_training: ${flag_training}"

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
if [ "${flag_training}" = "true" ]
then
    echo "-------------------start training model------------------------"
    if [ $USE_SFPS -eq 1 ]
    then
        IFS=","
        # arr default: 1,INFO,1,0,1,1,127.0.0.1,8135,lo,0,8,zmq,0,1
        # --thread-pool-size c num/2 default 8
        #arr=(${distribute_setting_no_space})
        echo "python ${train_workdir}/train/main.py $@"
        arr=(1 INFO 1 0 0 1 127.0.0.1 8135 lo 0 8 zmq 0 1)
        # arr=${sfps_param}
        export PYTHONUNBUFFERED=1
        export OMPI_ALLOW_RUN_AS_ROOT=1
        export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
        TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" \
        TF_CPP_MIN_VLOG_LEVEL=0 HOROVOD_WITH_TENSORFLOW=${arr[0]} NCCL_DEBUG=${arr[1]} \
        # mpilaunch --node-number ${arr[2]} --node-id ${arr[3]} --servers ${arr[4]} --workers ${arr[5]} --scheduler-ip ${arr[6]} --scheduler-port ${arr[7]} --interface ${arr[8]} \
        # --verbose ${arr[9]} \
        mpilaunch --node-number ${arr[2]} --node-id ${arr[3]} --servers ${arr[4]} --workers ${workers} --scheduler-ip ${arr[6]} --scheduler-port ${arr[7]} --interface ${arr[8]} \
        --verbose ${verbose} \
        --thread-pool-size ${arr[10]} \
        --env DMLC_ENABLE_RDMA:${arr[11]} \
        --env BYTEPS_ENABLE_IPC:${arr[12]} \
        --env SFPS_THREADS_NUM:${arr[13]} \
        --env SFPS_DEVICE_SEED:35243 \
        python ${train_workdir}/train/main.py $@ --use_sfps True \
        --config_task_file=${config_path} \
        --data_dir=${target_path} \
        --OUTPUT_DIR=${model_path} \
        --LOG_DIR=${log_path}
    else
        python -u ${code_path}/train/main.py ${day_end} \
        --config_task_file=${config_path} \
        --data_dir=${target_path} \
        --OUTPUT_DIR=${model_path} \
        --LOG_DIR=${log_path}
    fi

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

