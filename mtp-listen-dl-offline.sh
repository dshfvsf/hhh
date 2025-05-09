#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# detect latest available file and start training.
if  [ "${train_data_flag}" = "true" ]
then
  algorithm_dir=$algorithm_dirs
  origen_config_dir=/opt/huawei/schedule-train/algorithm/train.config
else
  algorithm_dir=/opt/huawei/schedule-train/algorithm
  origen_config_dir=$algorithm_dir/train.config
fi
config_DIR=$algorithm_dir/python/DL_algorithm/data_process/config_task.py
cp $origen_config_dir $config_DIR
common_feature=true
DAY_WINDOW=${DAY_WINDOW:-33}  # 训练数据窗口长度（天数），即day_start距离现在33天
allow_last_day=${allow_last_day:-'-5'}
echo "allow_last_day:$allow_last_day"
cur_day_flag=`date -u -d "${allow_last_day} days +8 hours" +%Y%m%d`  # 记录当前最新的day_end(初始值可修改)
echo "cur_day_flag:$cur_day_flag"
model_path=$OUTPUT_DIR
data_path=$data_dir
training_days=0
alternate_flag=false
incremental_flag=false
target_dir=$target_dir

incremental_flag_or_not=${incremental}
# "F" means full training, "I" means incremental training, and "FI" meand alternately training
if [ ! $incremental_flag_or_not ]
then
    incremental_flag=false
elif [ "$incremental_flag_or_not" = "F" ]
then
    incremental_flag=false
elif [ "$incremental_flag_or_not" = "I" ]
then
    incremental_flag=true
    alternate_flag=true
elif [ "$incremental_flag_or_not" = "FI" ]
then
    incremental_flag=false
    alternate_flag=true
fi
if [ ! $alternate_days_or_not ]; then
    alternate_days=-1
else
    alternate_days=$alternate_days_or_not
fi

echo $alternate_days

if [ "${incremental_flag}" = "true" ]
then
    DAY_WINDOW=2
fi
remove_day_flag=`date -u -d "-34 days +8 hours" +%Y%m%d`;
echo ${remove_day_flag}" data removed"

DAY_NUM=1;
target_day_flag=`date -u -d "-$DAY_NUM days +8 hours" +%Y%m%d`;
file_name=$target_day_flag/merge_rawId.txt

if test -e $target_dir/id_data;
then
    echo $target_dir/id_data
    rm -rf $target_dir/id_data
fi

if test -e $target_dir/featureMap.txt;
then
    echo $target_dir/featureMap.txt
    rm -rf $target_dir/featureMap.txt
fi

if test -e $target_dir/statistic.info;
then
    echo $target_dir/statistic.info
    rm -rf $target_dir/statistic.info
fi

if [ "${incremental_flag}" = "false" ] && [ -e $target_dir/model ];
then
   echo $target_dir/model
   rm -rf $target_dir/model
fi

if test -e $target_dir/bidata_attr;
then
    echo $target_dir/bidata_attr
    rm -rf $target_dir/bidata_attr
fi
if test -e $target_dir/size_file;
then
    echo $target_dir/size_file
    rm -rf $target_dir/size_file
fi


echo "file_name:${data_path}${file_name}"
echo "cur_day_flag:${cur_day_flag}"
echo "target_day_flag:${target_day_flag}"

if [ "${cur_day_flag}" != "${target_day_flag}" ] && [ -f ${data_path}${file_name} ]
then
    echo ${target_day_flag}" data detected, waiting for scp complete...";
    day_start=`date -u -d "-$DAY_WINDOW days +8 hours" +%Y%m%d`;
    if [ "${incremental_flag}" = "true" ]
    then
        echo "incremental training"
        last_file=$day_start/merge_rawId.txt
        while [ ! -f ${data_path}${last_file} ]
        do
            echo $last_file
            DAY_WINDOW=$(($DAY_WINDOW+1))
            day_start=`date -u -d "-$DAY_WINDOW days +8 hours" +%Y%m%d`;
            echo $DAY_WINDOW
            last_file=$day_start/merge_rawId.txt
        done
    fi
#     sleep 9;  # wait until data file complete scp
    echo "incremental and alternate " ${incremental_flag} ${alternate_flag}
    echo "alternately training days and count days" ${alternate_days} ${training_days}
    cur_day_flag=${target_day_flag};
    training_days=$(($training_days+1))
    echo "The time1 is: "
    echo $day_start
    echo ${target_day_flag}
    ckpt_day=${day_start}
    if [ "${incremental_flag}" = "true" ]
    then
        if [ "${baseModel_dir}" = "" ]
        then
            ckpt_dir=${target_dir}
        else
            ckpt_dir=${baseModel_dir}
        fi
        while [ ! -f ${ckpt_dir}"/model/${ckpt_day}_frozen_model.pb" ]
        do
            ckpt_day=`date -d "${ckpt_day} -1 day" +%Y%m%d`;
            echo "matching ckpt day:"${ckpt_day}
        done
    echo "final ckpt day:"${ckpt_day}
    fi
    if bash $algorithm_dir/python/DL_algorithm/mtp-run.sh ${day_start} ${target_day_flag} ${incremental_flag} ${config_DIR} ${common_feature} ${ckpt_day};then echo "训练OK";else exit 1;fi
    echo "finished training " ${target_day_flag}
elif [ $cur_day_flag ]
then
    while [ $target_day_flag -gt $cur_day_flag ]
    do
        file_name=$target_day_flag/merge_rawId.txt
        if [ -f ${data_path}${file_name} ]
        then
            echo ${target_day_flag}" data detected, waiting for scp complete...";
            day_start=`date -u -d "-$DAY_WINDOW days +8 hours" +%Y%m%d`;
#             sleep 90;  # wait until data file complete scp
            if [ "${incremental_flag}" = "true" ]
            then
                echo "incremental training"
                last_file=$day_start/merge_rawId.txt
                while [ ! -f ${data_path}${last_file} ]
                do
                    echo $last_file
                    DAY_WINDOW=$(($DAY_WINDOW+1))
                    day_start=`date -u -d "-$DAY_WINDOW days +8 hours" +%Y%m%d`;
                    echo $DAY_WINDOW
                    last_file=$day_start/merge_rawId.txt
                done
                DAY_WINDOW=$(($DAY_WINDOW+1))
                day_start=`date -u -d "-$DAY_WINDOW days +8 hours" +%Y%m%d`;
            fi
            echo "incremental and alternate " ${incremental_flag} ${alternate_flag}
            echo "alternately training days and count days" ${alternate_days} ${training_days}
            cur_day_flag=${target_day_flag};
            training_days=$(($training_days+1))
            echo "The time2 is: "
            echo $day_start
            echo ${target_day_flag}
            ckpt_day=${day_start}
            if [ "${incremental_flag}" = "true" ]
            then
                if [ "${baseModel_dir}" = "" ]
                then
                    ckpt_dir=${target_dir}
                else
                    ckpt_dir=${baseModel_dir}
                fi
                while [ ! -f ${ckpt_dir}"/model/${ckpt_day}_frozen_model.pb" ]
                do
                    ckpt_day=`date -d "${ckpt_day} -1 day" +%Y%m%d`;
                    echo "matching ckpt day:"${ckpt_day}
                done
            echo "final ckpt day:"${ckpt_day}
            fi

            if bash $algorithm_dir/python/DL_algorithm/mtp-run.sh ${day_start} ${target_day_flag} ${incremental_flag} ${config_DIR} ${common_feature} ${ckpt_day};then echo "训练OK";else exit 1;fi
            echo "finished training " ${target_day_flag}
            break;
        else
            echo ${target_day_flag}" data not arrived..."
            DAY_NUM=`expr $DAY_NUM + 1`  # detect data of one more day ago.
            target_day_flag=`date -u -d "-$DAY_NUM days +8 hours" +%Y%m%d`;
        fi
    done
fi