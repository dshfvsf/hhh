code_path=$3

gpu="0"
first_day_flag=$2
for day_end in $1
do
  day_start=${day_end}
  te_day=`date -d "${day_end} +1 day" +'%Y%m%d'`
  #python -u  ${code_path}/data_process/process_Bi_data.py ${day_start} ${te_day} --day_end_as_testset --incremental=True  --config_task_file=${code_path}/data_process/config_task.py
  python -u  ${code_path}/data_process/process_Bi_data.py ${day_start} ${day_start} --incremental=True  --config_task_file=${code_path}/data_process/config_task.py

  if [ $? -ne 0 ]; then
    exit 1
  fi

  echo "====== start train incremental model of [${day_start},${day_start}] ======"
  if [ ${first_day_flag} -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=${gpu} python -u ${code_path}/train/main.py ${day_start} --incremental --config_task_file=${code_path}/data_process/config_task.py --first_day_flag
    #first_day_flag=0
  else
    CUDA_VISIBLE_DEVICES=${gpu} python -u ${code_path}/train/main.py ${day_start} --incremental --config_task_file=${code_path}/data_process/config_task.py
  fi
  if [ $? -ne 0 ]; then
    exit 1
  fi
done
echo "finish all model evaluation on test data."
