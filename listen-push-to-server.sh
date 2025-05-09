#!/usr/bin/env bash
# -*- coding: utf-8 -*-


base_dir="/opt/huawei/data/topsearch/pushModel/"

while true
do
    cur_featureMap=`stat ${base_dir}"featureMap.txt" | grep -i Modify | awk -F. '{print $1}'|awk '{print $2" "$3}'`
    cur_model=`stat ${base_dir}"model.pb" | grep -i Modify | awk -F. '{print $1}'|awk '{print $2" "$3}'`


    if [ "${cur_featureMap}" == "" ] || [ "${cur_model}" == "" ]
    then
        echo "not all files ready, waiting..."
    else
        if [ "${prev_featureMap}" != "${cur_featureMap}" ] && [ "${prev_model}" != "${cur_model}" ] # all 2 files are updated
        then
            echo "new model, featureMap, applist ready."
            bash push-model-to-server.sh
            echo -e "finished pushing these files\nfeatureMap\t${cur_featureMap}\nmodel\t${cur_model}"
            prev_featureMap=${cur_featureMap}
            prev_model=${cur_model}
        else
            echo "waiting for update..."
        fi
    fi
    sleep 900
    echo ${prev_featureMap}
    echo ${prev_model}
done

