#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# detect latest available file and start training.
config_DIR="xxxxxxxxxxxxx"
common_feature=true
DAY_WINDOW=16  # 训练数据窗口长度（天数），即day_start距离现在15天
cur_day_flag=`date -u -d '-5 days +8 hours' +%Y%m%d`  # 记录当前最新的day_end(初始值可修改)
model_path="/home/hispace/container/data/topsearch/"
data_path=$model_path"train_data/"
training_days=0
alternate_flag=false
incremental_flag=false
train_with_push=false
mkdir $model_path/pushModel
# "F" means full training, "I" means incremental training, and "FI" meand alternately training
if [ ! $1 ]
then
    incremental_flag=false
elif [ "$1" = "F" ]
then
    incremental_flag=false
elif [ "$1" = "I" ]
then
    incremental_flag=true
    alternate_flag=true
elif [ "$1" = "FI" ]
then
    incremental_flag=false
    alternate_flag=true
fi
if [ ! $2 ]; then
    alternate_days=-1
else
    alternate_days=$2
fi
while true
do
    if [ "${incremental_flag}" = "true" ]
    then
        DAY_WINDOW=2
    fi
    remove_day_flag=`date -u -d "-34 days +8 hours" +%Y%m%d`;
    echo ${remove_day_flag}" data removed"
    if test -e $data_path/train_data_from_first_"${remove_day_flag}".001.txt;
    then
        rm -rf $data_path/train_data_from_first_"${remove_day_flag}".001.txt
    fi

    if test -e $data_path/train_data_from_first_"${remove_day_flag}".001.txt.preproc;
    then
        rm -rf $data_path/train_data_from_first_"${remove_day_flag}".001.txt.preproc
    fi

    if test -e $model_path/model/"${remove_day_flag}"-DCN_frozen_model.pb;
    then
        rm -rf $model_path/model/"${remove_day_flag}"*
    fi

    DAY_NUM=1;
    target_day_flag=`date -u -d "-$DAY_NUM days +8 hours" +%Y%m%d`;
    file_name="train_data_from_first_"${target_day_flag}".001.txt"
    if [ "${cur_day_flag}" != "${target_day_flag}" ] && [ -f ${data_path}${file_name} ]
    then
        echo ${target_day_flag}" data detected, waiting for scp complete...";
        day_start=`date -u -d "-$DAY_WINDOW days +8 hours" +%Y%m%d`;
        if [ "${incremental_flag}" = "true" ]
        then
            echo "incremental training"
            last_file="train_data_from_first_"${day_start}".001.txt"
            while [ ! -f ${data_path}${last_file} ]
            do
                echo $last_file
                DAY_WINDOW=$(($DAY_WINDOW+1))
                day_start=`date -u -d "-$DAY_WINDOW days +8 hours" +%Y%m%d`;
                echo $DAY_WINDOW
                last_file="train_data_from_first_"${day_start}".001.txt"
            done
        fi
        sleep 90;  # wait until data file complete scp
        echo "incremental and alternate " ${incremental_flag} ${alternate_flag}
        echo "alternately training days and count days" ${alternate_days} ${training_days}
        cur_day_flag=${target_day_flag};
        training_days=$(($training_days+1))
        if [ "${train_with_push}" = "false" ]
        then
            bash ./run.sh ${day_start} ${target_day_flag} ${incremental_flag} ${config_DIR} ${common_feature} > ./dl_${target_day_flag}.out;
        else
            bash ./run_training_and_push.sh ${day_start} ${target_day_flag} ${incremental_flag} ${config_DIR} ${common_feature} > ./dl_${target_day_flag}.out;
        fi
        echo "finished training " ${target_day_flag}
    elif [ $cur_day_flag ]
    then
        while [ $target_day_flag -gt $cur_day_flag ]
        do
            file_name="train_data_from_first_"${target_day_flag}".001.txt"
            if [ -f ${data_path}${file_name} ]
            then
                echo ${target_day_flag}" data detected, waiting for scp complete...";
                day_start=`date -u -d "-$DAY_WINDOW days +8 hours" +%Y%m%d`;
                sleep 90;  # wait until data file complete scp
                if [ "${incremental_flag}" = "true" ]
                then
                    echo "incremental training"
                    last_file="train_data_from_first_"${day_start}".001.txt"
                    while [ ! -f ${data_path}${last_file} ]
                    do
                        echo $last_file
                        DAY_WINDOW=$(($DAY_WINDOW+1))
                        day_start=`date -u -d "-$DAY_WINDOW days +8 hours" +%Y%m%d`;
                        echo $DAY_WINDOW
                        last_file="train_data_from_first_"${day_start}".001.txt"
                    done
                    DAY_WINDOW=$(($DAY_WINDOW+1))
                    day_start=`date -u -d "-$DAY_WINDOW days +8 hours" +%Y%m%d`;
                fi
                echo "incremental and alternate " ${incremental_flag} ${alternate_flag}
                echo "alternately training days and count days" ${alternate_days} ${training_days}
                cur_day_flag=${target_day_flag};
                training_days=$(($training_days+1))
                if [ "${train_with_push}" = "false" ]
                then
                    bash ./run.sh ${day_start} ${target_day_flag} ${incremental_flag} ${config_DIR} ${common_feature} > ./dl_${target_day_flag}.out;
                else
                    bash ./run_training_and_push.sh ${day_start} ${target_day_flag} ${incremental_flag} ${config_DIR} ${common_feature} > ./dl_${target_day_flag}.out;
                fi
                echo "finished training " ${target_day_flag}
                break;
            else
                echo ${target_day_flag}" data not arrived..."
                DAY_NUM=`expr $DAY_NUM + 1`  # detect data of one more day ago.
                target_day_flag=`date -u -d "-$DAY_NUM days +8 hours" +%Y%m%d`;
            fi
        done
    else
        echo ${target_day_flag}" data not arrived..."
    fi
    sleep 300
    # if you want incremental training after full training, you shoud set DAY_WINDOWN=2 and incremental_flag=true
    # DAY_WINDOW=2
    if [ $alternate_flag = "true" ] && [ $training_days -eq $alternate_days ]
    then
       incremental_flag=false 
       training_days=0
    elif [ $alternate_flag = "true" ] && [ $training_days -ne $alternate_days ]
    then
       incremental_flag=true
    fi
done

