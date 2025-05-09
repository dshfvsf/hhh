# 运行方式

> 注：业务侧在路径需要保存listen-dl-offline.sh,listen-push-to-server.sh,config_task.py,run.sh文件

程序使用监听方式运行

使用时分别后台执行训练和推送脚本

`nohup sh listen-dl-offline.sh`

`nohup sh listen-push-to-server.sh`

# 参数配置

> 参数都存放在配置文件config_task.py中

参数组成可以分为三类：

1.特征部分



| 参数            | 含义                                                         |
| --------------- | ------------------------------------------------------------ |
| REGRESSION_TASK | 是否是回归任务                                               |
| COLUMN_SPECS    | 特征信息，主要包含name（特征名）,to_use（是否使用）,sparse_feature（是否离散/稀疏特征）,to_preprocess（特殊处理方式）,feature_source（特征来源） |
|                 | 对于多任务模型，前N行均为LABEL，以2任务为例：                  |
|                 | COLUMN_SPECS = [{'name': 'LABEL1', 'to_use': True, 'sparse_feature':True, 'to_preprocess': 'convert_label1'}, |
|                 |                {'name': 'LABEL2', 'to_use': True, 'sparse_feature':True, 'to_preprocess': 'convert_label2'}, |
|                 |                {'name': 'App.ID', 'to_use': True, 'sparse_feature':True, 'feature_source': 'app'}, |
|                 |                {'name': 'App.Cat', 'to_use': True, 'sparse_feature':True, 'feature_source': 'app'}] |
| WIDE_FEATURES   | Wide 侧特征                                                  |
| ADD_COLUMN      | 额外默认特征                                                 |
| SHARE_INDEX     | 共享特征域（如downapp和useapp共享特征）                      |
| SHARE_ID        | 共享特征名（如榜单1和榜单2指同一个a榜单)                     |
| APP_ID_INDEX    | 交互式DCN 目标app id索引位置（暂无用）                       |
| DOWN_APP_INDEX  | 交互式DCN 用户下载app 索引位置（暂无用）                     |
| DYNAMIC_LENGTH  | 是否使用变长的多值特征                                       |
| INTERACTIVE_VARIABLE  | 交互式DCN 用户下载app 命名（暂无用）                   |

2.文件部分

| 参数                    | 含义                                 |
| ----------------------- | ------------------------------------ |
|TEST_DAY_NUM |测试集天数|
|TRIAN_DAY_INTERVAL|训练集距最后一天距离|
| RAW_FILE                | 原始文件名                           |
| BASE_DIR                | 数据地址                             |
| DATA_TYPE               | 数据类型，支持h5和tfrecord           |
| DATA_FROM               | 数据来源，支持原始数据和公共特征数据 |
| TRAIN_RECORD            | tfrecord训练路径及名字               |
| TEST_RECORD             | tfrecord测试路径及名字               |
| NEG_SAMP_RATIO          | 负采样率                             |
| RAND_SEED               | 随机种子                             |
| FLTR                    | 过滤阈值                             |
| APPID_FLTR              | 应用id过滤阈值                       |
| APP_CNT_THRESH          | 应用 候选过滤阈值                    |
| PAD_TO_LEN              | multi hot pad固定长度值              |
| DYNAMIC_PAD_TO_LEN      | multi hot pad变化长度值              |
| DELIMITER               | 数据间隔符                           |
| PREPROC_POSTFIX         | 预处理文件后缀                       |
| LINES_PER_PART          | 每个part文件行数                     |
| LINES_PER_PART_TFRECORD | 每个tfrecord part文件行数            |
| SAMPLES_PER_LINE        | tfrecord每行样本数                   |
| MULTI_HOT_DELIMITER     | multi_hot特征分隔符                      |
| MINIMUM_EPOCH | 最小运行轮数 |

3.模型部分

| 参数                       | 含义                                                   |
| -------------------------- | ------------------------------------------------------ |
| ALGORITHM                  | 使用的算法                                             |
| TRANSFER_MODE              | 迁移学习模式                                           |
| USE_CROSS                  | 是否使用Cross层                                        |
| USE_LINEAR                 | 是否使用线性层                                         |
| USE_FM                     | 是否使用FM层                                           |
| TRAIN_SCORE                | 每训练100批次打印日志                                  |
| DATA_SPLIT                 | 数据集切分，每轮使用1/n                                |
| BATCH_SIZE                 | 批次大小                                               |
| POS_WEIGHT                 | 无用，保持默认                                         |
| N_EPOCH                    | 轮数                                                   |
| EARLY_STOP_EPOCHS          | 早停轮数                                               |
| MODE                       | 执行训练或者测试                                       |
| EARLY_S                    | 早停分值（如为0.1，则本轮分值高于上轮0.1才计入早停）   |
|LOSS                        | 计算loss的类型，CROSS类型表示交叉熵计算loss，否则使用受限的PRAUC计算loss，只出现在DCN和SNR算法中
| EMBEDDING_SIZE             | 嵌入层大小                                             |
| NUM_CROSS_LAYER            | Cross层数                                              |
| DEEP_LAYERS                | 神经网络层节点数                                       |
| ACT_FUNC                   | 激活函数                                               |
| INIT_METHOD                | 初始化方式                                             |
| MIN_VALUE                  | 初始化分布最小值                                       |
| MAX_VALUE                  | 初始化分布最大值                                       |
| SEEDS                      | 随机种子                                               |
| INIT_PATH                  | 无用，保持默认                                         |
| OPTIMIZER                  | 优化器                                                 |
| LEARNING_RATE              | 学习率                                                 |
| EPSILON                    | 模糊因子                                               |
| DECAY_RATE                 | 衰减值                                                 |
| DECAY_STEPS                | 衰减步数                                               |
| KEEP_PROB                  | 随机失活率                                             |
| L2_LAMBDA                  | L2系数                                                 |
| L1_LAMBDA                  | L1系数                                                 |
| LOSS_MODE                  | 损失组合方式                                           |
| MERGE_MULTI_HOT            | 是否合并多值特征                                       |
| BATCH_NORM                 | 批正则                                                 |
| -------------------------- | **以下为非DCN部分，不使用无需关注** ------------------ |
| K                          | FFM 嵌入层大小                                         |
| LAMBDA                     | FFM loss系数                                           |
| REDUCE_BY_METHOD           | FFM loss计算方式                                       |
| INTERACTIVE                | 是否使用交互式DCN                                               |
| NUM_ATTENTION_LAYER        | attention层数                                          |
| ATTENTION_EMBEDDING_SIZE   | attention 嵌入层大小                                   |
| HEAD_NUM                   | attention 多头个数                                     |
| USE_RESIDUAL               | 是否使用残差连接                                       |
| DEEP_MERGE_MULTI           | wide&deep deep是否合并多值特征                         |
| WIDE_MERGE_MULTI           | wide&deep  wide是否合并多值特征                        |
|  DEEP_WIDE_FEATURE |   the list of feature type, True means Deep, False means Wide, e.g [True, False] |
| EMBED_LOSS_MODE            | loss计算方式                                           |
| WIDE_OPTIMIZER             | wide部分优化器                                         |
| WIDE_LEARNING_RATE         | wide部分学习率                                         |
| WIDE_ACC                   | wide累计值                                             |
| WIDE_LOSS_MODE             | wide损失模式                                           |
| REGULATION_LOSS            | 正则损失                                               |
| TEMPERATURE                | 蒸馏标签系数                                           |
| KD_LAMBDA                  | 蒸馏损失系数                                           |
| DISTILL                    | 是否蒸馏                                               |
| DAY_END_AS_TESTSET         | 是否最后一天作为测试集                                 |
| FIXED_EPOCH                | 是否训练固定轮数                                       |
| AUTOFIS_TOPK               | AutoFIS特征topK                                        |
| AUTOFIS_MASK               | AutoFIS mask保存numpy文件                              |
| WEIGHT_BASE                | AutoFIS 结构参数权重均值                               |
| WEIGHT_RANGE               | AutoFIS 结构参数权重范围                               |
| WEIGHT_SEED                | AutoFIS 结构参数权重随机种子                           |
| WEIGHT_L1                  | AutoFIS L1正则化系数                                   |
| WEIGHT_L2                  | AutoFIS L2正则化系数                                   |
| ITEM_COL                   | 多任务学习模型构造feature_map的定位标识符          |
| LABEL                      | 多任务学习模型LABEL数量                           |
| MULTI_TASK                   | 多任务学习模型标识符（True：使用多任务学习模型；False：使用非多任务学习模型；默认为False |
| NUM_EXPERT_UNITS           | MMOE专家核数                                     |
| NUM_EXPERTS                | MMOE专家数                                       |
| NUM_TASKS                  | MMOE任务数                                       |
| EXPERT_ACT_FUNC            | MMOE专家激活函数                                  |
| TOWER_DEEP_LAYERS          | MMOE塔深度网络层数及每层核数                       |
| TOWER_ACT_FUNC             | MMOE塔激活函数                                   |
|LOW_LAYER_UNIT_NUM           | SNR共享网络底层网络个数，默认值3                      |
|HIGH_LAYER_UNIT_NUM          | SNR共享网络上层网络个数，默认值3                  |
|Z_BETA                      | SNR共享网络转移参数计算元素，默认值0.6，取值区间[0.5,0.9]|
|Z_GAMMA                     |SNR共享网络转移参数计算元素，默认值-0.3，取值区间[-1,-0.1]|
|Z_TAO                       |SNR共享网络转移参数计算元素，默认值2，取值区间[1.1,2]|
| INIT_METHOD                | SNR z_alpha初始化方式，默认值uniform                                          |
| MIN_VALUE                  | SNR z_alpha初始化分布最小值，默认值-1                                     |
| MAX_VALUE | SNR z_alpha初始化分布最大值，默认值1 |
| -------MTP部分---- |  |
| MTP | 是否开启MTP模式 |
| OUPUT_DIR | 输出地址(model) |
| DATA_DIR | 数据地址 |
| LOG_DIR | 日志地址 |
| TARGET_DIR | 数据输出地址 |
| REMOVE_TIME | 数据删除周期 |
| DATA_KEEP_DAYS | 数据保留天数 |
| DATA_USE_DAYS | 使用数据天数 |
| PROCESS_NUM | 多进程数 |
| -------END------ |  |
|  |  |
