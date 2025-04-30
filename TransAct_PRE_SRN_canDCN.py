# -*- coding: utf-8 -*-
# @Time    : 2024/5/23
# @Author  :
# @File    : TransAct_PRE_SRN_canDCN.py
# @Overview:
from __future__ import print_function

from typing import Union, List

import copy
import numpy as np
import tensorflow as tf

from train.models.tf_util import pretrained_embedding, local_split_embedding, init_var_map, activate, loss_choose  # add
from train.layer.SRN import SRN  # add
from train.layer.MNS import MNS  # add
from train.layer.TransAct import TransAct  # add
from train.layer.DHEN import DHEN  # add
from train.layer.WuKong_V2 import WuKong  # add
from train.layer.MoMME import MoMME  # add
from train.layer.DCN_v3 import DeepCrossNetv3, ShallowCrossNetv3, MultiHeadFeatureEmbedding  # add
from train.layer.CoreLayer import JrcLayer, HierrecLayer
from train.layer.CoreLayer import HiNetLayer, DomainTower
from train.models.canDCN import canDCN

try:
    import horovod.tensorflow as hvd
except Exception:
    print("have no horovod package")
finally:
    pass


class TransActPreSRNcanDCN(canDCN):

    def __init__(self, config,
                 dataset_argv,
                 architect_argv,
                 init_argv,
                 ptmzr_argv,
                 reg_argv,
                 distilled_argv,
                 loss_mode='full',
                 merge_multi_hot=False,
                 batch_norm=True,
                 distill=False,
                 checkpoint=None,
                 sess=None,
                 regression_task=False,
                 use_cross=True,
                 use_linear=False,
                 one_hot_idx_list=None,
                 multi_hot_idx_list=None,
                 use_fm=False,
                 use_can=False,
                 can_as_mlp_input=False,
                 can_as_can_module=False,
                 dense_as_mlp_input=True,
                 dense_as_module=False
                 ):
        self.train_preds, self.eval_preds = None, None
        self.features_dim = None
        self.embed_v = None
        self.hash_table = None
        self.embed_linear = None
        self.posi_embed = None
        self.dnn_input_dim = None
        self.all_deep_layer = None
        self.h_w, self.h_b = None, None
        self.inf_dropout = None
        self.cross_input_dim = None
        self.cross_w = None
        self.cross_b = None
        self.hinet = None
        self.hierrec_layer = None
        self.jrc_cls = None
        self.final_output = None
        self.final_output_pos = None
        self.final_output_p = None
        #
        self.use_dnn = getattr(config, "USE_DNN", True)
        self.pretrained_embedding_config = None
        self.pretrained_embedding_list = []
        self.pre_emb_dict = {}
        self.srn_config = getattr(config, "srn_config", [])
        self.srn_var_list = []
        self.srn_use_orig_input = False
        self.domain_tower = None
        #
        _, _, _, multi_hot_flags, _, multi_hot_variable_len = dataset_argv
        self.config = config
        self.all_feature_config = getattr(self.config, "all_feature_config", {})
        self.semantic_id_init(init_argv, architect_argv[0])
        self.select_feature_index = None
        if not self.config.delete_idxs:
            dataset_argv[3], dataset_argv[5] = self.remove_feature_init(multi_hot_flags,
                                                                        multi_hot_variable_len
                                                                        )
        #
        self.use_pre_emb = getattr(config, 'USE_PRE_EMB', False)
        self.use_srn = getattr(config, 'USE_SRN', False)
        self.use_transAct = getattr(config, 'USE_TransAct', False)
        self.transAct = None
        self.use_dhen = getattr(config, 'USE_DHEN', False)
        self.dhen = None
        self.use_aux_cvr = getattr(config, 'USE_AUX_CVR', False)
        self.aux_cvr_h_w = []
        self.aux_cvr_h_b = []
        self.aux_cvr_out_w = None
        self.aux_cvr_out_b = None
        self.aux_cvr_final_output = None
        self.use_wukong = getattr(config, 'USE_WUKONG', False)
        self.wukong = None
        self.use_momme = getattr(config, 'USE_MOMME', False)
        self.momme = None
        self.use_dcnv3 = getattr(config, 'USE_DCNV3', False)
        if self.use_dcnv3:
            self.num_deep_cross_layers = getattr(self.config, "NUM_DEEP_CROSS_LAYERS", 4)
            self.num_shallow_cross_layers = getattr(self.config, "NUM_SHALLOW_CROSS_LAYERS", 4)
            self.deep_net_dropout = getattr(self.config, "DEEP_NET_DROPOUT", 0.1)
            self.shallow_net_dropout = getattr(self.config, "SHALLOW_NET_DROPOUT", 0.3)
            self.layer_norm = getattr(self.config, "DCN_V3_LAYER_NORM", True)
            self.batch_norm = getattr(self.config, "DCN_V3_BATCH_NORM", False)
            self.num_heads = getattr(self.config, "DCN_V3_NUM_HEADS", 1)
            self.multihead_embedding_layer = MultiHeadFeatureEmbedding(self.num_heads)
            architect_argv[0] *= self.num_heads
            self.dcnv3, self.scnv3 = None, None
            self.y_dcnv3, self.y_scnv3 = None, None
        ###################
        super(TransActPreSRNcanDCN, self).__init__(
                config,
                dataset_argv,
                architect_argv,
                init_argv,
                ptmzr_argv,
                reg_argv,
                distilled_argv,
                loss_mode,
                merge_multi_hot,
                batch_norm,
                distill,
                checkpoint,
                sess,
                regression_task,
                use_cross,
                use_linear,
                one_hot_idx_list,
                multi_hot_idx_list,
                use_fm,
                use_can,
                can_as_mlp_input,
                can_as_can_module,
                dense_as_mlp_input,
                dense_as_module
        )
    ###############add by##############

    def semantic_id_init(self, init_argv, embedding_size=90):
        self.semantic_embedding_dict = {}
        semantic_embedding_config = getattr(self.config, "semantic_embedding_config", [])
        if semantic_embedding_config:
            if isinstance(semantic_embedding_config, dict):
                semantic_embedding_config = [semantic_embedding_config]
            for semantic_emb_config in semantic_embedding_config:
                name = semantic_emb_config.get("name", "default")
                max_index = semantic_emb_config.get("max_index", 100)
                trainable = semantic_emb_config.get("trainable", False)
                dtype = semantic_emb_config.get("dtype", 'float32')
                if name not in self.semantic_embedding_dict:
                    if init_argv[0] == 'uniform':
                        low, high = init_argv[1:3]
                        embedding = np.random.uniform(low, high, size=(max_index, embedding_size))
                    else:  # use normal distribution
                        loc, scale = init_argv[1:3]
                        embedding = np.random.normal(loc, scale, size=(max_index, embedding_size))
                    if trainable:
                        self.semantic_embedding_dict[f'{name}_embed'] = tf.Variable(embedding, dtype=dtype)
                    else:
                        self.semantic_embedding_dict[f'{name}_embed'] = tf.constant(embedding, dtype=dtype)
            print(f"semantic_embedding_dict={self.semantic_embedding_dict}")

    def remove_feature_init(self, multi_hot_flags, multi_hot_variable_len):
        self.orig_multi_hot_variable_len = copy.deepcopy(multi_hot_variable_len)
        self.orig_multi_hot_flags = copy.deepcopy(multi_hot_flags)
        remove_feature_config = getattr(self.config, "remove_feature_config", {})
        if remove_feature_config:
            print(f"remove_feature_config={remove_feature_config}")
            self.select_feature_index = np.arange(len(multi_hot_flags))
            remove_feature_index = remove_feature_config.get("remove_feature_index", [])
            remove_multi_index = remove_feature_config.get("remove_multi_index", [])

            multi_hot_flags = np.delete(multi_hot_flags, remove_feature_index, axis=0)
            self.select_feature_index = np.delete(self.select_feature_index, remove_feature_index, axis=0)
            multi_hot_variable_len = [multi_hot_variable_len[idx]
                                      for idx in range(len(multi_hot_variable_len))
                                      if idx not in remove_multi_index]
        return multi_hot_flags, multi_hot_variable_len

    def pre_embedding_init(self, config):
        pretrained_embedding_config = getattr(config, "pretrained_embedding_config", [])
        self.pretrained_embedding_config = pretrained_embedding_config
        if pretrained_embedding_config:
            if isinstance(pretrained_embedding_config, dict):
                pretrained_embedding_config = [pretrained_embedding_config]
            self.pretrained_embedding_list = []
            for i, pre_emb_config in enumerate(pretrained_embedding_config):
                print(f"{i} pre_emb_config={pre_emb_config}")
                if 'init_argv' not in pre_emb_config:
                    pre_emb_config['init_argv'] = self.init_argv
                if 'feature_name' not in pre_emb_config:
                    pre_feat_nm = pre_emb_config['pre_feat_nm']
                    pre_emb_config['feature_name'] = self.all_feature_config[pre_feat_nm]['feature_name']
                self.pretrained_embedding_list.append(pretrained_embedding(**pre_emb_config))

    def pre_embedding_layer(self, input_id_hldr, cross_intput_list=None, dnn_input_list=None,
                            final_input_list=None, embed_output=None, training=False
                            ):
        if training:
            self.pre_embedding_init(self.config)
        self.pre_emb_dict = {}
        for i, pre_emb_config in enumerate(self.pretrained_embedding_config):
            if 'pre_id_index' in pre_emb_config:
                pre_emb = tf.gather(self.pretrained_embedding_list[i],
                                    tf.gather(input_id_hldr, pre_emb_config['pre_id_index'], axis=1)
                                    )
            else:
                feature_config = self.all_feature_config[pre_emb_config['pre_feat_nm']]
                if feature_config['is_remove']:
                    pre_emb = tf.gather(self.pretrained_embedding_list[i],
                                        tf.gather(self.id_hldr if training else self.eval_id_hldr,
                                                  feature_config['orig_index'], axis=1
                                                  )
                                        )
                else:
                    pre_emb = tf.gather(self.pretrained_embedding_list[i],
                                        tf.gather(input_id_hldr, feature_config['index'], axis=1)
                                        )
            semantic_name = pre_emb_config.get('semantic_name', None)
            if semantic_name and semantic_name in self.semantic_embedding_dict:
                pre_emb = tf.gather(self.semantic_embedding_dict[semantic_name], pre_emb)
                pre_emb = tf.reduce_mean(pre_emb, axis=-2)
                print(f"{semantic_name} pre_emb={pre_emb}")
            if tf.shape(pre_emb).shape == 3:
                mean_pre_emb = tf.reduce_mean(pre_emb, axis=1)
            else:
                mean_pre_emb = pre_emb
            add_type = pre_emb_config.get('add_type', "").split(",")
            if self.fields_num != 0 and self.use_dnn and 'add_dnn' in add_type:
                print(f"pre_emb add dnn")
                dnn_input_list.append(mean_pre_emb)
            if self.use_cross and 'add_cross' in add_type:
                print(f"pre_emb add cross")
                cross_intput_list.append(mean_pre_emb)
            if (self.use_fm and 'add_fm' in add_type) or 'add_input' in add_type:
                print(f"embed_output after={embed_output}")
                if embed_output is None:
                    embed_output = mean_pre_emb
                else:
                    embed_output = tf.concat([embed_output, mean_pre_emb], axis=1)
                pre_emb_dim = mean_pre_emb.shape.as_list()[-1]
                self.embed_output_dim += pre_emb_dim
                print(f"embed_output later={embed_output}")
            if 'add_final' in add_type:
                print(f"pre_emb add final")
                final_input_list.append(mean_pre_emb)
            if 'add_srn' in add_type:
                print(f"pre_emb add srn")
            name = pre_emb_config.get('name', pre_emb_config['pre_feat_nm'])
            self.pre_emb_dict[name] = pre_emb
        #
        print(f"pre_emb_dict={self.pre_emb_dict}")
        return embed_output

    def srn_init(self, config):
        self.srn_config = getattr(config, "srn_config", [])
        self.srn_var_list = []
        self.srn_use_orig_input = False
        for single_srn_config in self.srn_config:
            bin_type = single_srn_config.get("bin_type", 'SRN').upper()
            if bin_type == 'SRN':
                self.srn_var_list.append(SRN(init_argv=self.init_argv, **single_srn_config))
            elif bin_type == 'MNS':
                self.srn_var_list.append(MNS(init_argv=self.init_argv, **single_srn_config))
            if self.srn_use_orig_input:
                continue
            target_feat_nm = single_srn_config.get("target_feat_nm", None)
            sequence_feat_nm = single_srn_config.get("sequence_feat_nm", None)
            if target_feat_nm in self.all_feature_config and self.all_feature_config[target_feat_nm]['is_remove']:
                self.srn_use_orig_input = True
            if sequence_feat_nm in self.all_feature_config and self.all_feature_config[sequence_feat_nm]['is_remove']:
                self.srn_use_orig_input = True

    def srn_input(self, feat_nm, input_hot_v, orig_input_hot_v, idx=0):
        if feat_nm in self.all_feature_config and self.all_feature_config[feat_nm]['is_remove']:
            index_nm, mask_index_nm = 'orig_index', 'orig_mask_index'
            input_hot_v = orig_input_hot_v
        else:
            index_nm, mask_index_nm = 'index', 'mask_index'
        if feat_nm in self.pre_emb_dict:
            emb = self.pre_emb_dict[feat_nm]
        else:
            index = self.all_feature_config[feat_nm][index_nm]
            emb = tf.gather(input_hot_v[idx], index, axis=1)
        mask = None
        if self.all_feature_config[feat_nm]['feature_len'] > 1:
            index = self.all_feature_config[feat_nm][mask_index_nm]
            mask = tf.gather(input_hot_v[idx + 1], index, axis=1)
        return emb, mask

    def srn_layer(self, input_id_hldr, input_wt_hldr, cross_intput_list=None, dnn_input_list=None,
                  final_input_list=None, embed_output=None, training=False
                  ):
        if training:
            self.srn_init(self.config)
        if not self.srn_config:
            return embed_output
        input_hot_v = local_split_embedding(self.embed_v,
                                            input_id_hldr,
                                            input_wt_hldr,
                                            self.multi_hot_flags,
                                            self.multi_hot_variable_len
                                            )
        orig_input_hot_v = None
        if self.srn_use_orig_input:
            orig_input_hot_v = local_split_embedding(self.embed_v,
                                                     self.id_hldr if training else self.eval_id_hldr,
                                                     self.wt_hldr if training else self.eval_wt_hldr,
                                                     self.orig_multi_hot_flags,
                                                     self.orig_multi_hot_variable_len
                                                     )
        for idx, config_srn in enumerate(self.srn_config):
            print(f"srn_config[{idx}]={config_srn}")
            target_feat_nm = config_srn.get("target_feat_nm", None)
            target_emb, _ = self.srn_input(target_feat_nm, input_hot_v, orig_input_hot_v, 0)
            #
            sequence_feat_nm = config_srn.get("sequence_feat_nm", None)
            sequence_emb, sequence_mask = self.srn_input(sequence_feat_nm, input_hot_v, orig_input_hot_v, 2)
            srn_output = self.srn_var_list[idx].run([target_emb, sequence_emb], sequence_mask)
            add_type = config_srn.get('add_type', '').split(',')
            if self.use_dnn and 'add_dnn' in add_type:
                dnn_input_list.append(srn_output)
            if self.use_cross and 'add_cross' in add_type:
                cross_intput_list.append(srn_output)
            if self.use_fm and 'add_fm' in add_type:
                if embed_output is None:
                    embed_output = srn_output
                else:
                    embed_output = tf.concat([embed_output, srn_output], axis=1)
                self.embed_output_dim += self.embedding_size
            if 'add_final' in add_type:
                final_input_list.append(srn_output)
        return embed_output

    def get_input_hldr(self, training=True):
        input_dense_hldr = None
        if training:
            if self.select_feature_index is not None:
                input_id_hldr = tf.gather(self.id_hldr, self.select_feature_index, axis=1)
                input_wt_hldr = tf.gather(self.wt_hldr, self.select_feature_index, axis=1)
            else:
                input_id_hldr = self.id_hldr
                input_wt_hldr = self.wt_hldr

            if self.dense_num != 0 and self.use_dense_features:
                input_dense_hldr = self.dense_hldr
        else:
            if self.select_feature_index is not None:
                input_id_hldr = tf.gather(self.eval_id_hldr, self.select_feature_index, axis=1)
                input_wt_hldr = tf.gather(self.eval_wt_hldr, self.select_feature_index, axis=1)
            else:
                input_id_hldr = self.eval_id_hldr
                input_wt_hldr = self.eval_wt_hldr
            if self.dense_num != 0 and self.use_dense_features:
                input_dense_hldr = self.eval_dense_hldr
        return input_id_hldr, input_wt_hldr, input_dense_hldr

    def trans_act_init(self, seq_len):
        self.transAct = TransAct(self.config, self.init_argv, seq_len)

    def get_input_embedding(self,
                            input_id_hldr,
                            input_wt_hldr,
                            trans_act_feat_nm_list: list,
                            index=0,
                            training=False
                            ):
        remove_feature_names = getattr(self.config, "remove_feature_config", {'remove_feature_names': []}
                                       )['remove_feature_names']
        embedding_list = []
        seq_len = 0
        orig_input_hot_v, input_hot_v = None, None
        orig_mask, mask = None, None
        for trans_act_feat_nm, trans_act_pre_emb_name in trans_act_feat_nm_list:
            if trans_act_feat_nm in remove_feature_names:
                if orig_input_hot_v is None:
                    orig_input_hot_v, orig_mask = local_split_embedding(self.embed_v,
                                                                        self.id_hldr if training else self.eval_id_hldr,
                                                                        self.wt_hldr if training else self.eval_wt_hldr,
                                                                        self.orig_multi_hot_flags,
                                                                        self.orig_multi_hot_variable_len
                                                                        )[index: index + 2]
                    print(f"orig_input_hot_v={orig_input_hot_v}")
                item_index = self.all_feature_config[trans_act_feat_nm]['orig_index']
                if index == 2:
                    num_onehot = sum([not flag for flag in self.orig_multi_hot_flags])
                    item_index = [idx - num_onehot for idx in item_index]
            else:
                if input_hot_v is None:
                    input_hot_v, mask = local_split_embedding(self.embed_v,
                                                              input_id_hldr,
                                                              input_wt_hldr,
                                                              self.multi_hot_flags,
                                                              self.multi_hot_variable_len
                                                              )[index: index + 2]
                    print(f"input_hot_v={input_hot_v}")
                if index == 2:
                    item_index = self.all_feature_config[trans_act_feat_nm]['mask_index']
                else:
                    item_index = self.all_feature_config[trans_act_feat_nm]['index']
            print(f"{trans_act_feat_nm}.index={item_index}")
            if trans_act_pre_emb_name in self.pre_emb_dict:
                embedding = self.pre_emb_dict[trans_act_pre_emb_name]
            else:
                embedding = tf.gather(orig_input_hot_v if trans_act_feat_nm in remove_feature_names
                                      else input_hot_v, item_index, axis=1
                                      )
                mask = tf.gather(orig_mask if trans_act_feat_nm in remove_feature_names
                                 else mask, item_index, axis=1
                                 )
                embedding = tf.multiply(embedding, mask)
            shape = embedding.shape.as_list()
            if len(shape) == 3:
                seq_len += shape[1]
            embedding_list.append(embedding)
        return tf.concat(embedding_list, axis=1), seq_len

    def trans_act_run(self,
                      input_id_hldr,
                      input_wt_hldr,
                      item_embedding: tf.Tensor = None,
                      item_embedding_seq: Union[tf.Tensor, List[tf.Tensor]] = None,
                      action_type_seq: Union[tf.Tensor, List[tf.Tensor]] = None,
                      action_time_seq: tf.Tensor = None,
                      request_time: tf.Tensor = None,
                      training=False
                      ):
        if item_embedding is None:
            item_embedding, _ = self.get_input_embedding(input_id_hldr,
                                                         input_wt_hldr,
                                                         getattr(self.config, "trans_act_item_nm_list"),
                                                         0,
                                                         training
                                                         )
        item_embedding = tf.squeeze(item_embedding, axis=1)
        #
        if item_embedding_seq is None:
            item_embedding_seq, seq_len = self.get_input_embedding(input_id_hldr,
                                                                   input_wt_hldr,
                                                                   getattr(self.config, "trans_act_seq_nm_list"),
                                                                   2,
                                                                   training
                                                                   )
        else:
            seq_len = item_embedding_seq.shape.as_list()[1]
        if training:
            self.trans_act_init(seq_len)
        ###
        if action_type_seq is None:
            if isinstance(item_embedding_seq, list):
                type_seq = []
                for i, item_emb_seq in enumerate(item_embedding_seq):
                    type_seq.append(tf.ones_like(item_emb_seq[:, :, 0], dtype=tf.int64) * (i + 1))  # [None, seq_len]
                action_type_seq = tf.concat(type_seq, axis=-1)
            else:
                action_type_seq = tf.ones_like(item_embedding_seq[:, :, 0], dtype=tf.int64)  # [None, seq_len]
        if isinstance(action_type_seq, list):
            action_type_seq = tf.concat(action_type_seq, axis=-1)  # [None, seq_len*n]
        print(f"item_embedding={item_embedding}")
        print(f"item_embedding_seq={item_embedding_seq}")
        print(f"action_type_seq={action_type_seq}")
        trans_act_output = self.transAct.run(item_embedding,
                                             item_embedding_seq,
                                             action_type_seq,
                                             action_time_seq,
                                             request_time,
                                             training
                                             )
        print(f"trans_act_output={trans_act_output}")
        return trans_act_output

    def aux_cvr_layer(self, training=True):
        aux_cvr_config = getattr(self.config, "aux_cvr_config", {})
        user_nm = aux_cvr_config["user_nm"]
        item_nm = aux_cvr_config["item_nm"]
        # 获取训练user和item的embedding，并拼接起来
        mlp_output = tf.concat([self.pre_emb_dict.get(user_nm, None), self.pre_emb_dict.get(item_nm, None)], axis=-1)
        mlp_output = tf.squeeze(mlp_output, axis=1)
        if training:
            # MLP Layer的网络参数
            deep_layers = aux_cvr_config.get("deep_layers", [self.embedding_size])
            dnn_input_dim = mlp_output.shape.as_list()[1]
            aux_cvr_deep_layer = [dnn_input_dim] + deep_layers
            #
            init_acts = []
            for i in range(len(aux_cvr_deep_layer) - 1):
                init_acts.extend([('aux_cvr_h%d_w' % (i + 1), aux_cvr_deep_layer[i: i + 2], 'random'),
                                  ('aux_cvr_h%d_b' % (i + 1), [aux_cvr_deep_layer[i + 1]], 'random')])
            # Predict Layer的网络参数
            init_acts.extend([('aux_cvr_out_w', [aux_cvr_deep_layer[-1], 1], 'random'),
                              ('aux_cvr_out_b', [1], 'zero')])
            var_map, log = init_var_map(self.init_argv, init_acts)
            self.log += log
            #
            for i in range(len(aux_cvr_deep_layer) - 1):
                self.aux_cvr_h_w.append(tf.Variable(var_map['aux_cvr_h%d_w' % (i + 1)]))
                self.aux_cvr_h_b.append(tf.Variable(var_map['aux_cvr_h%d_b' % (i + 1)]))
            self.aux_cvr_out_w = tf.Variable(var_map['aux_cvr_out_w'])
            self.aux_cvr_out_b = tf.Variable(var_map['aux_cvr_out_b'])
        for i, (h_w, h_b) in enumerate(zip(self.aux_cvr_h_w, self.aux_cvr_h_b)):
            mlp_output = tf.matmul(activate(self.act_func, mlp_output), h_w) + h_b
            if self.batch_norm:
                print("setting bn for training stage")
                mlp_output = tf.layers.batch_normalization(mlp_output,
                                                           training=training,
                                                           reuse=not training,
                                                           name="aux_cvr_bn_%d" % i)
            if training:
                mlp_output = tf.nn.dropout(mlp_output, keep_prob=self.keep_prob)
        #
        hidden_output = tf.matmul(activate(self.act_func, mlp_output), self.aux_cvr_out_w) + self.aux_cvr_out_b
        self.aux_cvr_final_output = tf.reshape(hidden_output, [-1], name="aux_cvr_final_output")
        return mlp_output

    def momme_layer(self, ctr_final_output, input_id_hldr, training=True):
        momme_config = getattr(self.config, "momme_config", {})
        print(f"momme_config={momme_config}")
        target_nm = momme_config["target_nm"]
        user_seq_nm = momme_config["user_seq_nm"].split(",")
        if training:
            self.momme = MoMME(**momme_config)
        target_emb = tf.stop_gradient(self.pre_emb_dict.get(target_nm, None))
        if len(user_seq_nm) > 1:
            user_seq_emb = tf.concat([tf.stop_gradient(self.pre_emb_dict.get(seq_nm, None))
                                      for seq_nm in user_seq_nm], axis=1)
        else:
            user_seq_emb = tf.stop_gradient(self.pre_emb_dict.get(user_seq_nm[0], None))
        item_index = self.all_feature_config.get(target_nm, {'index': None})['index']
        item_id = None
        if item_index:
            item_id = tf.gather(input_id_hldr, item_index[0], axis=1)
        momme_output = self.momme(ctr_final_output, target_emb, user_seq_emb, item_id, training)
        print(f"momme_output={momme_output}")
        return momme_output

    def dcn_v3(self, vx_embed, training):
        if training:
            if self.num_deep_cross_layers > 0:
                self.dcnv3 = DeepCrossNetv3(num_cross_layers=self.num_deep_cross_layers,
                                            net_dropout=self.deep_net_dropout,
                                            layer_norm=self.layer_norm,
                                            batch_norm=self.batch_norm,
                                            num_heads=self.num_heads,
                                            return_x=False)
            if self.num_shallow_cross_layers > 0:
                self.scnv3 = ShallowCrossNetv3(num_cross_layers=self.num_shallow_cross_layers,
                                               net_dropout=self.shallow_net_dropout,
                                               layer_norm=self.layer_norm,
                                               batch_norm=self.batch_norm,
                                               num_heads=self.num_heads,
                                               return_x=False)
        vx_embed = self.multihead_embedding_layer(vx_embed)
        logit, dlogit, slogit = None, None, None
        if self.num_deep_cross_layers > 0:
            y_tmp_d = self.dcnv3(vx_embed, training)  # [B, H, 1]
            dlogit = tf.reshape(tf.reduce_mean(y_tmp_d, axis=1), [-1])
            logit = dlogit
        if self.num_shallow_cross_layers > 0:
            y_tmp_s = self.scnv3(vx_embed, training)  # [B, H, 1]
            slogit = tf.reshape(tf.reduce_mean(y_tmp_s, axis=1), [-1])
            logit = slogit if logit is None else (logit + slogit) * 0.5
        print(f"dcnv3 logit,dlogit,slogit={(logit, dlogit, slogit)}")
        return logit, dlogit, slogit

    def loss_part(self):
        super().loss_part()
        # add aux_cvr loss
        if self.use_aux_cvr:
            aux_cvr_loss = loss_choose(self.config, self.aux_cvr_final_output, self.lbl_hldr)
            aux_cvr_loss = tf.reduce_mean(aux_cvr_loss, name='loss')
            print(f"self.loss={self.loss}, aux_cvr_loss={aux_cvr_loss}")
            self.loss += aux_cvr_loss
            self.loss_dict['aux_cvr_loss'] = aux_cvr_loss
        # add scnv3 loss
        if self.use_dcnv3 and self.y_dcnv3 is not None and self.y_scnv3 is not None:
            loss_dcnv3 = tf.reduce_mean(loss_choose(self.config, self.y_dcnv3, self.lbl_hldr), name='loss_dcnv3')
            loss_scnv3 = tf.reduce_mean(loss_choose(self.config, self.y_scnv3, self.lbl_hldr), name='loss_scnv3')
            weight_dcnv3 = loss_dcnv3 - self.loss_dict["main_loss"]
            weight_scnv3 = loss_scnv3 - self.loss_dict["main_loss"]
            weight_dcnv3 = tf.where(weight_dcnv3 > 0, weight_dcnv3, 0.0)
            weight_scnv3 = tf.where(weight_scnv3 > 0, weight_scnv3, 0.0)
            loss_dcnv3 *= weight_dcnv3
            loss_scnv3 *= weight_scnv3
            print(f"self.loss={self.loss}, loss_dcnv3,loss_scnv3={(loss_dcnv3, loss_scnv3)}")
            self.loss += loss_dcnv3 + loss_scnv3
            self.loss_dict['loss_dcnv3'] = loss_dcnv3
            self.loss_dict['loss_scnv3'] = loss_scnv3
    ##############################################

    def build_model(self, training=True):
        cross_intput_list = []
        dnn_input_list = []
        final_input_list = []

        input_id_hldr, input_wt_hldr, input_dense_hldr = self.get_input_hldr(training)

        hinet_y = None
        embed_output = None
        if self.fields_num != 0:
            embed_output = self.get_embedding_component(input_id_hldr, input_wt_hldr, training)
            if self.use_dnn:
                dnn_input_list.append(embed_output)
            if self.use_cross:
                cross_intput_list.append(embed_output)

            if self.use_can:
                can_embed_output_list = self.get_can_embedding_component(input_id_hldr, input_wt_hldr, training)
                if self.can_as_mlp_input:
                    dnn_input_list.append(can_embed_output_list)
                    if self.use_cross:
                        cross_intput_list.append(can_embed_output_list)
                elif self.can_as_can_module:
                    final_input_list.append(can_embed_output_list)

            if self.config.use_hinet:
                if training:
                    self.hinet = HiNetLayer(self.config, self.num_multihot, self.num_onehot,
                                            self.embed_output_dim, self.init_argv
                                            )
                hinet_y = self.hinet(embed_output, input_id_hldr, training=training)

        if self.dense_num != 0 and \
                self.dense_as_mlp_input and self.use_dense_features:
            dnn_input_list.append(input_dense_hldr)

        ###############add##############
        if self.use_pre_emb:
            embed_output = self.pre_embedding_layer(input_id_hldr, cross_intput_list, dnn_input_list,
                                                    final_input_list, embed_output, training
                                                    )
        if self.use_srn:
            embed_output = self.srn_layer(input_id_hldr, input_wt_hldr, cross_intput_list, dnn_input_list,
                                          final_input_list, embed_output, training
                                          )
        if self.use_transAct:
            trans_act_output = self.trans_act_run(input_id_hldr, input_wt_hldr, training=training)
            if self.use_cross:
                cross_intput_list.append(trans_act_output)
            if self.use_dnn:
                dnn_input_list.append(trans_act_output)
        if self.use_dhen:
            if training:
                dhen_config = getattr(self.config, "dhen_config", {})
                self.dhen = DHEN(self.init_argv, **dhen_config)
            feat_num = int(embed_output.shape.as_list()[-1] / self.embedding_size)
            dhen_output = self.dhen.run(tf.reshape(embed_output, [-1, feat_num, self.embedding_size]),
                                        training)
            dhen_output = tf.layers.flatten(dhen_output)
            print(f"dhen_output={dhen_output}")
            final_input_list.append(dhen_output)
        if self.use_aux_cvr:
            aux_cvr_mlp_output = self.aux_cvr_layer(training)
            aux_cvr_mlp_output = tf.stop_gradient(aux_cvr_mlp_output)
            print(f"aux_cvr_mlp_output={aux_cvr_mlp_output}")
            if self.use_dnn:
                dnn_input_list.append(aux_cvr_mlp_output)
            if self.use_cross:
                cross_intput_list.append(aux_cvr_mlp_output)
        if self.use_wukong:
            wukong_config = getattr(self.config, "wukong_config", {})
            if training:
                feat_num = int(embed_output.shape.as_list()[-1] / self.embedding_size)
                wukong_config['init_argv'] = self.init_argv
                wukong_config['num_features'] = feat_num
                wukong_config['embedding_dim'] = self.embedding_size
                print(f"wukong_config={wukong_config}")
                self.wukong = WuKong(**wukong_config)
            wukong_output = self.wukong(embed_output, training=training)
            print(f"wukong_output={wukong_output}")
            if wukong_config.get("add_dnn", False):
                dnn_input_list.append(wukong_output)
            elif wukong_config.get("add_cross", False):
                cross_intput_list.append(wukong_output)
            elif wukong_config.get("add_final", False):
                final_input_list.append(wukong_output)
        if self.use_dcnv3:
            feat_num = int(embed_output.shape.as_list()[-1] / self.embedding_size)
            final_output, self.y_dcnv3, self.y_scnv3 = self.dcn_v3(
                    tf.reshape(embed_output, [-1, feat_num, self.embedding_size]),
                    training)
        ##############################################

        # cross module
        if len(cross_intput_list) != 0:
            cross_intput = tf.concat(cross_intput_list, axis=1)
            cross_output = self.get_cross_component(cross_intput, training)
            final_input_list.append(cross_output)

        # dnn module
        if len(dnn_input_list) != 0:
            dnn_input = tf.concat(dnn_input_list, axis=1)
            dnn_output = self.get_dnn_component(dnn_input, training)
            final_input_list.append(dnn_output)

        spe_logits = None
        if self.config.use_hierrec:
            if training:
                self.hierrec_layer = HierrecLayer(self.config, self.init_argv, self.log)
            hierrec_out = self.hierrec_layer(embed_output, dnn_output, training)
            final_input_list.append(hierrec_out)
            if self.config.use_spe_feature:
                spe_input = tf.reshape(embed_output, [-1, self.num_multihot + self.num_onehot, self.embedding_size])
                spe_logits = self.hierrec_layer.spe_feature_part(spe_input, training)

        if self.use_linear:
            linear_ouput = self.get_linear_component(input_id_hldr,
                                                     input_wt_hldr,
                                                     one_hot_idx_list=self.one_hot_idx_list,
                                                     multi_hot_idx_list=self.multi_hot_idx_list,
                                                     training=training
                                                     )
            final_input_list.append(linear_ouput)

        if self.use_fm:
            fm_output = self.get_fm_component(embed_output, training)
            final_input_list.append(fm_output)

        if self.dense_num != 0 \
                and self.dense_as_module and self.use_dense_features:
            dense_output = tf.reduce_sum(input_dense_hldr, axis=1, keep_dims=True)
            final_input_list.append(dense_output)

        # final module
        ctr_output = None
        if len(final_input_list) != 0:
            final_input = tf.concat(final_input_list, axis=1)
            if self.use_pos:
                if self.use_price:
                    ctr_output, price_output, pos_output = self.get_tower_component(final_input, training)
                    final_output, final_output_p, final_output_pos = self.get_final_component(ctr_output, pos_output,
                                                                                              price_output, training
                                                                                              )
                else:
                    ctr_output, _, pos_output = self.get_tower_component(final_input, training)
                    final_output, _, final_output_pos = self.get_final_component(ctr_output, pos_output, None, training)
            else:
                final_output, _, _ = self.get_final_component(final_input, training=training)
            if self.config.use_jrc:
                if training:
                    self.jrc_cls = JrcLayer(self.config, self.init_argv)
                _logit = ctr_output if self.use_pos else final_input
                final_output = self.jrc_cls.get_jrc_logit(_logit, final_output, training)

            if self.config.use_hinet:
                final_output += hinet_y

            if self.use_domain_tower:
                if training:
                    self.domain_tower = DomainTower(self.config, self.num_multihot, self.num_onehot,
                                                    self.embedding_size, self.init_argv, use_final=True
                                                    )
                domain_bias = self.domain_tower(embed_output, input_id_hldr, training)
                final_output += domain_bias

            if self.config.use_spe_feature:
                final_output += spe_logits

        if self.use_momme:
            final_output = self.momme_layer(final_output, input_id_hldr, training)
        if training:
            self.final_output = final_output
            if self.use_pos:
                self.final_output_pos = final_output_pos
            if self.use_price:
                self.final_output_p = final_output_p
            if self.regression_task:
                self.train_preds = tf.identity(final_output, name='predicitons')
            else:
                self.train_preds = tf.sigmoid(final_output, name='predicitons')
        else:
            if self.regression_task:
                self.eval_preds = tf.identity(final_output, name='predictionNode')
            else:
                self.eval_preds = tf.sigmoid(final_output, name='predictionNode')
