echo "=== data arrived, running run-dl-offline.sh ==="
# 1. process data
algorithm_dir=$ALGO_DIR
day_start=$1
day_end=$2
incremental_code=$3
config_DIR=$4
common_feature=$5
ckpt_day=$6
docker_name=python2
day_end_as_testset=true
model_path=$OUTPUT_DIR
code_path=$algorithm_dir/python/DL_algorithm
data_path=$data_dir
target_path=$target_dir
LOG_DIR=$LOG_DIR

echo $algorithm_dir
echo $data_dir
echo $target_dir
echo $OUTPUT_DIR

if  [ "${common_feature}" = "true" ]
then
  if python -u $code_path/data_process/process_Bi_data_common_feature.py ${day_start} ${day_end} --day_end_as_testset --incremental=${incremental_code}  --config_task_file=${config_DIR} --data_dir=$data_path --target_dir=$target_path;then echo "预处理OK";else exit 1;fi
else
  if python -u $code_path/data_process/process_Bi_data.py ${day_start} ${day_end} --day_end_as_testset --incremental=${incremental_code}  --config_task_file=${config_DIR} --data_dir=$data_path --target_dir=$target_path;then echo "预处理OK";else exit 1;fi
fi

echo "finish processing data from "${day_start}" to "${day_end}

# 2. train model
if  [ "${DISTRIBUTE}" = "true" ]
then
  echo $target_dir
elif  [ "${incremental_code}" = "true" ]
then
  if python -u $code_path/train/main.py ${day_end} --incremental --config_task_file=${config_DIR} --OUTPUT_DIR=$model_path --data_dir=$target_dir --baseModel_dir=$baseModel_dir --LOG_DIR=$LOG_DIR --checkpoint_day=${ckpt_day};then echo "训练OK";else exit 1;fi
else
  if python -u $code_path/train/main.py ${day_end} --config_task_file=${config_DIR} --OUTPUT_DIR=$model_path --data_dir=$target_dir --LOG_DIR=$LOG_DIR;then echo "训练OK";else exit 1;fi
fi