echo "=== data arrived, running run_AutoFIS_template.sh ==="
# 1. process data
day_start=$1
day_end=$2
config_DIR=$3
common_feature=$4
docker_name=python2
day_end_as_testset=true
gpu="0"
model_path=/opt/huawei/data/ads/game_autofis/
code_path=/opt/huawei/common/algorithm/python/DL_algorithm

if [ ! -d "$model_path/pushModel" ]; then
  mkdir $model_path/pushModel
fi
docker exec -i $docker_name /bin/sh -c "
if  [ "${common_feature}" = "true" ]
then
  python -u $code_path/data_process/process_Bi_data_common_feature.py ${day_start} ${day_end} --day_end_as_testset --config_task_file=${config_DIR}
else
  python -u $code_path/data_process/process_Bi_data.py ${day_start} ${day_end} --day_end_as_testset --config_task_file=${config_DIR}
fi
";
echo "finish processing data from "${day_start}" to "${day_end}

# 2. train model & cp model & featuremap
docker exec -i $docker_name /bin/sh -c "
CUDA_VISIBLE_DEVICES=${gpu} python -u $code_path/train/main.py ${day_end}  --search_or_retrain='search' --config_task_file=${config_DIR}
echo "finish-autofis-search-stage"
CUDA_VISIBLE_DEVICES=${gpu} python -u $code_path/train/main.py ${day_end}  --search_or_retrain='retrain' --config_task_file=${config_DIR}
echo "finish-autofis-retrain-stage"
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

