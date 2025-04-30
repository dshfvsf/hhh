import os
import argparse
import tensorflow as tf
from models.MMOE_ALL import MMOE_ALL

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="data dir ")
args, unknown = parser.parse_known_args()
feature_map_dir = os.path.join(args.data_dir, 'model')
FEATURE_MAP_FILE = 'featureMap.txt'


class MMOE_ALL_LOSS(MMOE_ALL):
    def __init__(self, config, dataset_argv, embedding_size, expert_argv, tower_argv,
                 init_argv, ptmzr_argv, reg_argv, dense_flags, regression_task=False,
                 trans_argv=None, u_init_argv=None, loss_mode='full', merge_multi_hot=False, batch_norm=True):
        self.loss = None
        super().__init__(config, dataset_argv, embedding_size, expert_argv, tower_argv,
                         init_argv, ptmzr_argv, reg_argv, dense_flags, regression_task=False,
                         trans_argv=None, u_init_argv=None, loss_mode='full', merge_multi_hot=False, batch_norm=True)

    def get_target_feature_id(self, feature_mapdir):
        res_id = []
        feature_map_path = os.path.join(feature_mapdir, FEATURE_MAP_FILE)

        with open(feature_map_path, 'r', encoding='utf-8') as f:
            feature_map_list = f.readlines()
        feature_map_list = list(map(lambda x: x.strip(), feature_map_list))
        column = self.config.column
        list_id_list = self.config.list_id_list
        collum_list_id = []
        for i, _ in enumerate(list_id_list):
            temp_list = []
            for second_id in list_id_list[i]:
                temp_list.append(column[i] + ',' + second_id)
            collum_list_id.append(temp_list)

        for collum_list_id_i in collum_list_id:
            temp_list_id = []
            for list_id_i in collum_list_id_i:
                for feature_map in feature_map_list:
                    if list_id_i == feature_map.split('\t')[0]:
                        temp_list_id.append(int(feature_map.split('\t')[1]))
                        continue
            res_id.append(temp_list_id)
        return res_id

    def loss_part(self, final_layer_y, ptmzr_argv):
        target_id_list = self.get_target_feature_id(feature_map_dir)
        self.loss_weight_variable()
        self.loss = 0
        column_num_list = self.config.column_num_list

        group_id_list = []
        for i, column_num_i in enumerate(column_num_list):
            list_id_column = tf.gather(self.id_hldr, [column_num_i], axis=1)
            target_id_tensor = tf.constant(target_id_list[i], dtype=tf.int64)
            group_id_mask = tf.cast(
                tf.equal(list_id_column, target_id_tensor), dtype=tf.float32
            )
            if len(target_id_list) > 1:
                group_id_mask = tf.expand_dims(
                    tf.reduce_sum(group_id_mask, axis=-1), axis=-1
                )
            group_id_list.append(group_id_mask)

        mask_list = []
        loss_target_weight = self.config.loss_target_weight

        for j, mask_i in enumerate(loss_target_weight):
            loss_mask = group_id_list[j] * mask_i
            mask_list.append(loss_mask)


        for k, y_final in enumerate(final_layer_y):
            loss_i_target = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_final, labels=self.lbl_values[:, k])  # [B]

            for loss_weight_z in mask_list:
                loss_i_target += tf.multiply(loss_i_target, loss_weight_z)

            if self.has_task_mask:
                loss_i_target = tf.multiply(loss_i_target, self.lbl_masks[:, k])
            if self.mean_all_samples:
                loss_i_target = tf.reduce_mean(loss_i_target)
            else:
                num_samples = tf.reduce_sum(self.lbl_masks[:, k])
                num_samples = tf.maximum(num_samples, tf.constant(1.0))
                loss_i_target = tf.reduce_sum(loss_i_target) / num_samples
            self.loss += loss_i_target

        self.loss += tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_w[0]) \
                     + tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_b[0]) \
                     + tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_w[1]) \
                     + tf.contrib.layers.l1_regularizer(self.l1_lambda)(self.cross_b[1]) \
                     + self._lambda * tf.nn.l2_loss(self.embed_v)