echo "=== data arrived, running run-dl-offline.sh ==="
# 1. process data
day_start=$1
day_end=$2
incremental_code=$3
config_DIR=$4
common_feature=$5
docker_name=python2
day_end_as_testset=true
gpu="0"
model_path=/opt/huawei/data/ART/
code_path=/opt/huawei/common/algorithm/python/DL_algorithm

if [ ! -d "$model_path/pushModel" ]; then
  mkdir $model_path/pushModel
fi
docker exec -i $docker_name /bin/sh -c "
if  [ "${common_feature}" = "true" ]
then
  python -u $code_path/data_process/process_Bi_data_common_feature.py ${day_start} ${day_end} --day_end_as_testset --incremental=${incremental_code}  --config_task_file=${config_DIR}
else
  python -u $code_path/data_process/process_Bi_data.py ${day_start} ${day_end} --day_end_as_testset --incremental=${incremental_code}  --config_task_file=${config_DIR}
fi
";
echo "finish processing data from "${day_start}" to "${day_end}

docker exec -i $docker_name /bin/sh -c "
# 2. train model
if  [ "${incremental_code}" = "true" ]
then
  CUDA_VISIBLE_DEVICES=${gpu} python -u $code_path/train/main.py ${day_end} --incremental --config_task_file=${config_DIR}
else
  CUDA_VISIBLE_DEVICES=${gpu} python -u $code_path/train/main.py ${day_end} --config_task_file=${config_DIR}
fi
# 3. cp model & featuremap
if  [ "${common_feature}" = "true" ]
then
    cp $model_path/train_data/feature_map/feature_map.${day_end} $model_path/pushModel/featureMap.txt
else
    cp $model_path/model/featureMap.txt $model_path/pushModel/featureMap.txt
fi
cp $model_path/model/frozen_model.pb $model_path/pushModel/model.pb
chmod 777 $model_path/pushModel/featureMap.txt
chmod 777 $model_path/pushModel/model.pb
";