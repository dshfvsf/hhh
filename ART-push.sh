#!/usr/bin/env bash
# -*- coding: utf-8 -*-

export data_path=/opt/huawei/data/ART-MTP/train_data
export exec_data=/opt/huawei/hispace/ART/feature-engineering/execHive
export mtp_path="/opt/huawei/WiseCloudMTPTool_1.0.22.100/bin/start.sh"
CURR_PATH=$(cd "$(dirname "$mtp_path")"; pwd)
APP_DIR=$(dirname $CURR_PATH)
MODULE=mtp-push-tool
DATASET_NAME=gameoho
echo $APP_DIR
function status {
      CURR_PID=`ps -ef|grep java|grep $MODULE|grep ${APP_DIR}|grep -v grep|awk '{print $2}'`
      if  [ X"$CURR_PID" == X"" ];then
          return 1;
      else
          return 0;
      fi
}

# 取n天数据
function pull_push()
{
    DAY_NUM=$1
    now_day=`date -u -d "-0 days" +%Y%m%d`
    cur_day=`date -u -d "-$DAY_NUM days" +%Y%m%d`
    while [ $DAY_NUM -ge 0 ]
    do
        echo $cur_day"-0000" 
        bash $exec_data/exec-hive.sh ART.sh $cur_day"-0000"
        DAY_NUM=`expr $DAY_NUM - 1`
        echo $cur_day"'s data got"
        echo $DAY_NUM
        cur_day=`date -u -d "-$DAY_NUM days" +%Y%m%d`
    
    done
    
    # MTP 推送 
    dataSetId=$(grep "success, and the dataSetId is" $APP_DIR/logs/run.log | tail -1);
    echo "dataSedId" ${dataSetId:0-37:36};
    bash /opt/huawei/WiseCloudMTPTool_1.0.22.100/bin/start.sh -c append -f $data_path  -n $DATASET_NAME -b appstore -i ${dataSetId:0-37:36}
    # 判断是否此文件推送完成
    status
    while [ $? -eq 0 ]
    do
        echo "INFO" "$MODULE is aready running with pid=$CURR_PID"
        sleep 10
        status
    done
    rm -r $data_path/*
    return 0
}
if [ ! -d $data_path ];then
    mkdir $data_path
fi
echo "empty file for init" > $data_path/for_init
bash /opt/huawei/WiseCloudMTPTool_1.0.22.100/bin/start.sh -c upload -f $data_path  -n $DATASET_NAME -b appstore
echo "init upload done"
pull_push 1
while true
do
    pull_push 0
    sleep 12h
done
