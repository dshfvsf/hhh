#!/usr/bin/env bash
# -*- coding: utf-8 -*-

echo "===install the requirement==="

algorithm_dir=/opt/huawei/schedule-train/algorithm
echo "${algorithm_dir}"
pkg_dir=$algorithm_dir/DL_algorithm/package
req_file=$algorithm_dir/DL_algorithm/requirements.txt
# pip install --no-index --find-links=$pkg_dir -r $req_file --user


# detect latest available file and start training.
searchspace_dir=$algorithm_dir/DL_algorithm/autoreg/searchspace.json
tuner_dir=$algorithm_dir/DL_algorithm/autoreg
code_dir=$algorithm_dir/DL_algorithm/train
yaml_template=$tuner_dir/config_template.yaml
yaml_dir=$tuner_dir/config.yaml

origen_config_dir=$algorithm_dir/train.config
config_DIR=$algorithm_dir/DL_algorithm/data_process/config_task.py
cp $origen_config_dir $config_DIR
common_feature=true
echo $cur_day_flag
model_path=$OUTPUT_DIR
data_path=$data_dir
training_days=0
alternate_flag=false
incremental_flag=false
target_dir=$target_dir


# day_start=20210215
# target_day_flag=20210221



# day_start=20210215
# day_end=20210221

day_start=$day_start
day_end=$day_end

#auc_func_day=20210221
auc_func_day=$auc_func_day

incremental_code=false
docker_name=python2
day_end_as_testset=true
code_path=$algorithm_dir/DL_algorithm
LOG_DIR=$LOG_DIR

echo "generate config yaml"
if  [ "${DISTRIBUTE}" = "true" ]
then
  echo $target_dir
elif  [ "${incremental_code}" = "true" ]
then
python -u $code_path/autoreg/gen_yaml.py ${day_end} --incremental --yaml_file ${yaml_template} --searchspace ${searchspace_dir} --tuner ${tuner_dir} --code ${code_dir} --config_task=${config_DIR} --OUTPUT_DIR=$model_path --data_dir=$target_dir --LOG_DIR=$LOG_DIR
else
python -u $code_path/autoreg/gen_yaml.py ${day_end} --yaml_file ${yaml_template} --searchspace ${searchspace_dir} --tuner ${tuner_dir} --code ${code_dir} --config_task=${config_DIR} --OUTPUT_DIR=$model_path --data_dir=$target_dir --LOG_DIR=$model_path
fi

echo "run the autoreg"
python -u $code_path/autoreg/run.py --config ${yaml_dir}