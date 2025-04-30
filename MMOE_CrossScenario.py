from __future__ import print_function

import json
from train.models.tf_util import STARMLP
import tensorflow as tf
try:
    import horovod.tensorflow as hvd
    distribute_tmp = True
except Exception:
    print("have no horovod package")
    distribute_tmp = False
from models.MMOE_ALL import MMOE_ALL


class ConditionalBatchNormalization:
    def __init__(self, num_features, scope="conditional_bn") -> None:
        self.scope = scope
        self.num_features = num_features
        self.gamma = tf.Variable(tf.ones([num_features]), name=f'{scope}_gamma')
        self.beta = tf.Variable(tf.zeros([num_features]), name=f'{scope}_beta')

    def forward(self, inputs, delta_beta, delta_gamma, training=False):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            inputs = tf.layers.batch_normalization(
                inputs,
                center=False,
                scale=False,
                training=training,
                name='inner_bn'
            )
            input_shape = inputs.get_shape().as_list()
            pad_dim = [1] * (len(input_shape)-1)
            delta_beta = tf.reshape(delta_beta, [-1] + pad_dim)
            delta_beta = tf.tile(delta_beta, pad_dim + [self.num_features])   # [B, *, D]
            beta = tf.reshape(self.beta, pad_dim + [-1])  # [1, *, D]
            beta = tf.repeat(beta, tf.shape(inputs)[0], axis=0)
            beta = beta + delta_beta   # [B, *, D]

            delta_gamma = tf.reshape(delta_gamma, [-1] + pad_dim)
            delta_gamma = tf.tile(delta_gamma, pad_dim + [self.num_features])   # [B, *, D]
            gamma = tf.reshape(self.gamma, pad_dim + [-1])  # [1, *, D]
            gamma = tf.repeat(gamma, tf.shape(inputs)[0], axis=0)
            gamma = gamma + delta_gamma   # [B, *, D]

            return gamma * inputs + beta





class MMOE_CrossScenario(MMOE_ALL):

    def __init__(self, config, dataset_argv, embedding_size, expert_argv, tower_argv,
                 init_argv, ptmzr_argv, reg_argv, dense_flags, regression_task=False,
                 trans_argv=None, u_init_argv=None, loss_mode='full', merge_multi_hot=False,
                 batch_norm=True):
        self.scenario_id_index = None
        self.num_feat = None
        self.scenario_id_map = None
        self.scenario_gid_map = None
        self.num_scenario = None
        self.feature_map = dataset_argv[-1]
        self.feat2domain_net_hidden = [1024, 512]
        self.enable_domain_emb = False
        self.domain_emb_dim = 512
        self.enable_domain_gate = False
        self.stop_grad_global_domain_emb = False
        self.enable_star = False
        self.star_expert_dict = None
        self.domain_dict = {}
        self.embedding_dim = None
        self.loss = None
        self.domain_embs = None
        self.domain_cl_loss = 0.0

        self.enable_task_emb = False
        self.task_embs = None
        self.task_emb_dim = 512
        self.feat2task_net_hidden = [1024, 512]
        self.task_cl_loss = 0.0
        self.stop_grad_global_task_emb = False

        dataset_argv = dataset_argv[:-1]
        super(MMOE_CrossScenario, self).__init__(
            config, dataset_argv, embedding_size, expert_argv, tower_argv,
            init_argv, ptmzr_argv, reg_argv, dense_flags, regression_task,
            trans_argv, u_init_argv, loss_mode, merge_multi_hot, batch_norm
        )

    
    def scenario_parameter_init(self):
        # assume all onehot features are in front of multi-hot features
        scenario_id_name = getattr(self.config, 'SCENARIO_ID_NAME', 'Context.ListID')

        use_fine_grained_list = getattr(self.config, "USE_FINE_GRAINED_LIST", True)

        if use_fine_grained_list:
            # use fine grained list, each list id is a domain
            scenario_id_list = []
            feature_map = self.feature_map # DataFrame
            for fv, raw_id in zip(feature_map.feature_value, feature_map.id):
                if scenario_id_name in fv:
                    scenario_id_list.append(raw_id)
                    
            scenario_id_map = {raw_id: new_id for new_id, raw_id in enumerate(scenario_id_list)}

            tmp = tf.fill((self.features_size + 1,), tf.constant(-1, dtype=tf.int64))   # 0 for padding
            self.scenario_id_map = tf.tensor_scatter_nd_update(
                        tmp, 
                        tf.reshape(
                            tf.constant(list(scenario_id_map.keys()), dtype=tf.int64),
                            (-1, 1)
                            ), 
                        tf.constant(list(scenario_id_map.values()), dtype=tf.int64),
                        name='scenario_id_map'
                    )
            self.num_scenario = len(scenario_id_map)

        else:
            # disable fine grained list, list ids in domain_dict value is a domain
            scenario_id_map = {}
            for domain_id, (d, vs) in enumerate(self.domain_dict.items()):
                for v in vs:
                    scenario_id_map[v] = domain_id
            tmp = tf.fill((self.features_size + 1,), tf.constant(-1, dtype=tf.int64))   # 0 for padding
            self.scenario_id_map = tf.tensor_scatter_nd_update(
                tmp, 
                tf.reshape(
                    tf.constant(list(scenario_id_map.keys()), dtype=tf.int64),
                    (-1, 1)
                ), 
                tf.constant(list(scenario_id_map.values()), dtype=tf.int64),
                name='scenario_id_map'
            )
            self.num_scenario = len(scenario_id_map.values())

        if self.enable_domain_emb:
            print("Initialize domain embeddings.")
            self.domain_embs = tf.Variable(
                tf.random.uniform(
                    shape=(self.num_scenario, self.domain_emb_dim),
                    minval=-0.001,
                    maxval=0.001,
                ),
                name="Domain_embedding"
            )

        print(f'num_scenario: {self.num_scenario}')
        print(f'scenario id map: {scenario_id_map}')
    

    def parameter_init(self, config, dataset_argv, embedding_size,
                       reg_argv, expert_argv, tower_argv, trans_argv,
                       merge_multi_hot=False):
        super(MMOE_CrossScenario, self).parameter_init(
            config, dataset_argv, embedding_size,
            reg_argv, expert_argv, tower_argv, trans_argv,
            merge_multi_hot
        )
        self.scenario_id_index = self.config.domain_col_idx
        self.enable_domain_gate = self.config.ENABLE_DOMAIN_GATE
        self.enable_domain_emb = self.config.ENABLE_DOMAIN_EMB
        self.feat2domain_net_hidden = self.config.FEAT2DOMAIN_NET_HIDDEN
        self.num_feat = self.embedding_dim // embedding_size
        self.domain_dict = self.config.domain_dict
        self.enable_star = self.config.ENABLE_STAR
        self.domain_emb_dim = self.config.DOMAIN_EMB_DIM
        self.stop_grad_global_domain_emb = self.config.STOP_GRAD_GLOBAL_DOMAIN_EMB
        self.stop_grad_global_task_emb = self.config.STOP_GRAD_GLOBAL_TASK_EMB
        
        # task embeddings
        self.enable_task_emb = self.config.ENABLE_TASK_EMB
        self.task_emb_dim: int = self.config.TASK_EMB_DIM
        self.feat2task_net_hidden: list = self.config.FEAT2TASK_NET_HIDDEN
        
        if self.scenario_id_index >= self.num_feat:
            raise ValueError(f"There are {self.num_feat} features in total, "
                             f"while got `SCENARIO_ID_INDEX` as {self.scenario_id_index}.")
        if self.enable_domain_gate:
            self.embedding_dim = (self.num_feat - 1) * embedding_size   # remove domain id
            self.scenario_parameter_init()
        if self.enable_star:
            self.star_expert_init()
        
        if self.enable_task_emb:
            self.task_emb_init()

        extra_argv = {
            "scenario_id_index": self.scenario_id_index,
            "enhance_domain_feat": self.config.ENHANCE_DOMAIN_FEAT,
            "enable_domain_emb": self.enable_domain_emb,
            "feat2domain_net_hidden": self.feat2domain_net_hidden,
            "domain_emb_dim": self.domain_emb_dim,
            "enable_domain_gate": self.enable_domain_gate,
            "domain_dict": self.domain_dict,
            "enable_star": self.enable_star,
            "enable_task_emb": self.enable_task_emb,
            "feat2task_net_hidden": self.feat2task_net_hidden,
            "task_emb_dim": self.task_emb_dim,
            "stop_grad_global_domain_emb": self.stop_grad_global_domain_emb,
            "stop_grad_global_task_emb": self.stop_grad_global_task_emb,
            "domain_lambda": self.config.AUX_LAMBDA,
            "task_lambda": self.config.AUX_LAMBDA_TASK,
            "domain_emb_mix": getattr(self.config, "DOMAIN_EMB_MIX", False),
            "enable_cbn": getattr(self.config, "ENABLE_CBN", False),
            "mcn_hidden_layers": getattr(self.config, "MCN_HIDDEN_LAYERS", [1024, 512]),
            "use_fine_grained_list": getattr(self.config, "USE_FINE_GRAINED_LIST", True)
        }
        self.log += json.dumps(extra_argv, default=str, sort_keys=True, indent=4)


    def star_expert_init(self):
        layers = [self.embedding_dim] + self.expert_hidden_layers
        domain_expert_dict = {}
        valid_domain_names = ['shared'] + list(self.domain_dict.keys())
        for sce_name in valid_domain_names:
            domain_expert_dict[sce_name] = []
            for j in range(self.num_experts):
                mlp_j = STARMLP(
                    layers=layers,
                    act_func="relu",
                    bias=True,
                    batch_norm=True,
                    scope=f"{sce_name}_expert_mlp_{j}"
                )
                domain_expert_dict[sce_name].append(mlp_j)
        self.star_expert_dict = domain_expert_dict

    
    def task_emb_init(self):
        self.task_embs = tf.Variable(
            tf.random.uniform(
                shape=(self.num_tasks, self.task_emb_dim),
                minval=-0.001,
                maxval=0.001,
            ),
            name=f'Task_embs'
        )

    
    def task_feat_generator(self, x, scope: str = "task_feature_generator"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            hidden = tf.stop_gradient(x)
            layers = self.feat2task_net_hidden + [self.task_emb_dim]
            for i, dim in enumerate(layers):
                if i != len(layers) - 1:
                    activation = 'relu'
                else:
                    activation = None
                hidden = tf.layers.dense(
                    hidden,
                    dim,
                    activation=activation,
                    use_bias=True, 
                    name=f'mlp_{i}'
                )
            features = hidden
            return features
        

    def cal_task_cl_loss(self, gen_task_feat, label):
        logits = tf.matmul(gen_task_feat, tf.transpose(self.task_embs))
        probs = tf.nn.softmax(logits, axis=-1)  # [B, N_task]
        pos_probs = tf.reduce_sum(tf.multiply(probs, label), axis=-1)
        pos_probs = tf.where(tf.less_equal(pos_probs, 0), tf.ones_like(pos_probs), pos_probs)
        return tf.reduce_mean(-tf.log(pos_probs + 1e-10))
    

    def cal_task_cl_loss_v2(self, gen_task_feat, label):
        task_emb_instance = tf.matmul(tf.transpose(label), gen_task_feat)   # [T, D]
        deno = tf.reduce_sum(label, axis=0) + 1e-10
        deno = tf.expand_dims(deno, axis=-1)
        task_emb_instance = task_emb_instance / deno
        sim_mat = tf.matmul(task_emb_instance, tf.transpose(self.task_embs))    # [T, T]
        sim_mat = tf.exp(sim_mat)
        pos_sim = tf.diag(sim_mat)
        neg_sim_sum = tf.reduce_sum(sim_mat, axis=-1)
        cl_loss = tf.reduce_mean(tf.log(neg_sim_sum + 1e-10) - tf.log(pos_sim + 1e-10))
        return cl_loss


    def domain_feature_generator(self, x, scope: str = "domain_feature_generator"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            print("Enable domain feature enhancement.")
            hidden = tf.stop_gradient(x)
            layers = self.feat2domain_net_hidden + [self.domain_emb_dim]
            for i, dim in enumerate(layers):
                if i != len(layers) - 1:
                    activation = 'relu'
                else:
                    activation = None
                hidden = tf.layers.dense(
                    hidden,
                    dim,
                    activation=activation,
                    use_bias=True, 
                    name=f'mlp_{i}'
                )
            features = hidden

            return features


    def cal_domain_cl_loss(self, features, list_id):
        pred = tf.nn.softmax(
            tf.matmul(features, tf.transpose(self.domain_embs)),
            axis=-1
        )
        list_id_label = tf.gather(self.scenario_id_map, list_id)
        loss = tf.reduce_mean(-tf.log(tf.gather(pred, list_id_label, axis=-1)))
        return loss
    

    def split_domain_emb(self, nn_input):
        embedding_size = self.embedding_dim // (self.num_feat - 1)
        # [B, NxD] -> [B, N, D]
        nn_input = tf.reshape(nn_input, [-1, self.num_feat, embedding_size])
        domain_emb = nn_input[:, self.scenario_id_index, :]
        if self.scenario_id_index == self.num_feat:
            # the scenerio_id is put at the last of all features
            other_emb = nn_input[:, :self.scenario_id_index, :]
        else:
            other_embs = [nn_input[:, :self.scenario_id_index, :], nn_input[:, self.scenario_id_index+1:, :]]
            other_emb = tf.concat(other_embs, axis=1)
        other_emb = tf.reshape(other_emb, [-1, self.embedding_dim])
        return domain_emb, other_emb


    def _fuse_domain_embs(self, domain_emb: tf.Tensor, enhanced_domain_emb=None, list_id=None):
        if enhanced_domain_emb is not None:
            if self.enable_domain_emb:
                domain_id = tf.gather(self.scenario_id_map, list_id)
                domain_emb_global = tf.gather(self.domain_embs, domain_id)
                if self.stop_grad_global_domain_emb:
                    domain_emb_global = tf.stop_gradient(domain_emb_global)
                if self.config.DOMAIN_EMB_MIX:
                    ele_dot_feat = tf.multiply(enhanced_domain_emb, domain_emb_global)
                    ele_add_feat = enhanced_domain_emb + domain_emb_global
                    enhanced_domain_emb = tf.concat(
                        [enhanced_domain_emb, domain_emb_global, ele_dot_feat, ele_add_feat],
                        axis=-1
                    )
                else:
                    enhanced_domain_emb = tf.concat([enhanced_domain_emb, domain_emb_global], axis=-1)
                domain_emb = tf.concat([domain_emb, enhanced_domain_emb], axis=-1)
            else:
                domain_emb = enhanced_domain_emb
        return domain_emb
    

    def epnet(self, domain_emb: tf.Tensor, input_emb: tf.Tensor, scope: str = "EPNet", 
              enhanced_domain_emb=None, list_id=None):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            domain_emb = self._fuse_domain_embs(
                domain_emb=domain_emb,
                enhanced_domain_emb=enhanced_domain_emb,
                list_id=list_id
            )
            in_emb = domain_emb
            n_dim = in_emb.get_shape().as_list()[-1]
            input_emb_dim = input_emb.get_shape().as_list()[-1]
            domain_emb_ = tf.layers.dense(
                in_emb,
                n_dim // 4, 
                activation="relu",
                use_bias=True, 
                name=f'mlp_1'
            )
            delta = tf.layers.dense(
                domain_emb_, 
                input_emb_dim, 
                activation="sigmoid",
                use_bias=True, 
                name=f'mlp_2'
            )
            out = tf.multiply(delta * 2, input_emb)
        return out
    

    def get_domain_id(self, training):
        print("Get domain index.")
        id_hldr = self.id_hldr if training else self.eval_id_hldr
        one_hot_flags = [not flag for flag in self.multi_hot_flags]
        one_hot_id = tf.transpose(
            tf.boolean_mask(tf.transpose(id_hldr, [1, 0]), one_hot_flags),
            [1, 0]
        )
        return one_hot_id[:, self.scenario_id_index]
        

    def expert_forward(
            self,
            nn_input,
            delta_gamma_beta,
            domain_id,
            num_experts,
            batch_norm,
            training,
            scope="MMOE_expert"
        ):
        if self.enable_star:
            print("Expert in: ", nn_input)
            domain_gid = tf.gather(self.scenario_gid_map, domain_id)
            indices = tf.argsort(domain_gid, axis=-1, direction="ASCENDING", stable=True, name="argsort_domain_gid")
            indices = tf.argsort(indices, axis=-1, direction="ASCENDING", stable=True, name="argsort_domain_indices")
            out = []
            for gid, g_domain in enumerate(self.domain_dict.keys()):
                mask = tf.equal(domain_gid, (tf.zeros_like(domain_gid)+gid))
                nn_input_g = tf.boolean_mask(nn_input, mask=mask)
                experts: STARMLP = self.star_expert_dict.get(g_domain, None)
                if experts is None:
                    raise ValueError(f"There is no domain-specific experts for {g_domain}")
                exp_out = []
                for i, expert in enumerate(experts):
                    try:
                        shared_expert_var = self.star_expert_dict['shared'][i].var_dict
                    except KeyError:
                        shared_expert_var = None
                    x = expert.forward(
                        nn_input_g,
                        shared_fc_var=shared_expert_var,
                        training=training
                    )
                    exp_out.append(x)
                exp_out = tf.stack(exp_out, axis=1) # b x N_e x D
                out.append(exp_out)
            out = tf.concat(out, axis=0) # B x N_e x D
            out = tf.gather(out, indices, axis=0)   # B x N_e x D
        else:
            if delta_gamma_beta is not None:
                delta_gamma_beta = tf.reshape(delta_gamma_beta, [-1, len(self.expert_hidden_layers), 2])
                with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                    shared_expert_output_list = []
                    for i in range(num_experts):
                        shared_expert_output_list.append(
                            self.mlp_module(nn_input,
                                            delta_beta=delta_gamma_beta[:, :, 0],
                                            delta_gamma=delta_gamma_beta[:, :, 1],
                                            hidden_layers=self.expert_hidden_layers,
                                            act_func='relu', 
                                            dropout_prob=(1.0 - self.keep_prob),
                                            batch_norm=batch_norm, 
                                            training=training, 
                                            reuse=tf.AUTO_REUSE,
                                            name_scope=f'expert_{i}_mlp')) # [B, H_E]
                    out = tf.stack(shared_expert_output_list, axis=-2) # [B, N_E, H_E]
            else:
                out = super(MMOE_CrossScenario, self).expert_forward(
                    nn_input=nn_input,
                    num_experts=num_experts,
                    batch_norm=batch_norm,
                    training=training,
                    scope="MMOE_expert"
                )
        return out

    
    def calculate_expert_gate(
            self, 
            nn_input, 
            batch_norm,
            gate_act_func,
            num_experts, num_tasks,
            training
        ):
        # first forward: input -> experts
        delta_gamma_beta = None
        enhance_feat = None
        domain_index = None
        if self.enable_domain_gate:
            domain_emb, input_emb = self.split_domain_emb(nn_input)
            list_id = self.get_domain_id(training)
            if self.config.ENHANCE_DOMAIN_FEAT:
                enhance_feat = self.domain_feature_generator(input_emb)
                if training:
                    self.domain_cl_loss = self.cal_domain_cl_loss(enhance_feat, list_id)
            nn_input = self.epnet(domain_emb, input_emb, scope="EPNet", 
                                enhanced_domain_emb=enhance_feat, 
                                list_id=list_id)
            if getattr(self.config, "ENABLE_CBN", False):
                delta_gamma_beta = self.meta_condition_net(
                    domain_emb=domain_emb,
                    enhanced_domain_emb=enhance_feat,
                    list_id=list_id
                )
        else:
            nn_input = nn_input

        # first forward: input -> experts
        if self.input_gate_method == 'trm':
            print('using trm gate')
            nn_input_list = []
            for i in range(num_tasks):
                refine_output = self.task_refinement_network(nn_input, scope=f"task_{i}_refine_net")
                nn_input_list.append(tf.multiply(nn_input, refine_output))
            nn_input = tf.stack(nn_input_list, axis=1)  # [B, num_tasks, D]\
        elif self.input_gate_method == 'task_mask_gate':
            print('using task mask gate')
            nn_input = self.mask_refinement_component(nn_input)
        elif self.input_gate_method == 'single_gate':
            print('using single gate')
            refine_output = self.task_refinement_network(nn_input, 
                                                         hidden_factor=self.hidden_factor,
                                                         scope=f'single_task_refine_net')
            nn_input = tf.multiply(nn_input, refine_output)
        else:
            nn_input = nn_input

        shared_expert_output = self.expert_forward(
            nn_input=nn_input,
            delta_gamma_beta=delta_gamma_beta,
            domain_id=domain_index,
            num_experts=num_experts,
            batch_norm=batch_norm,
            training=training,
            scope="MMOE_expert"
        )   # [B, num_tasks, N_E, H_E] or [B, N_E, H_E]

        # second forward: input -> gate value
        gate_outputs = self.cal_gate_output(
            nn_input=nn_input,
            num_tasks=num_tasks,
            gate_act_func=gate_act_func,
            batch_norm=batch_norm,
            training=training
        )   # [B, N_task, N_E]
        
        bottom_outputs = self.syn_expert_with_gate(
            expert_output=shared_expert_output, 
            gate_output=gate_outputs
        )    # [N_task, B, H_E]
        return bottom_outputs
    

    def mlp_module(self, nn_input, hidden_layers, act_func='relu', 
                   dropout_prob=0.1, batch_norm=False, inf_dropout=False, 
                   delta_beta=None, delta_gamma=None,
                   training=True, reuse=True, name_scope='mlp_module'):
        if getattr(self.config, "ENABLE_CBN", False) and \
            (delta_beta is not None) and \
            (delta_gamma is not None):
            return self._mlp_module_cbn(
                nn_input, delta_beta, delta_gamma, hidden_layers, act_func, 
                dropout_prob, batch_norm, inf_dropout, training, reuse, name_scope
            )
        else:
            return super(MMOE_CrossScenario, self).mlp_module(
                nn_input, hidden_layers, act_func, dropout_prob, 
                batch_norm, inf_dropout, training, reuse, name_scope
            )
        

    def meta_condition_net(self, domain_emb: tf.Tensor, enhanced_domain_emb=None, list_id=None):
        domain_emb = self._fuse_domain_embs(
            domain_emb=domain_emb,
            enhanced_domain_emb=enhanced_domain_emb,
            list_id=list_id
        )
        num_expert_layers = len(self.expert_hidden_layers)
        with tf.variable_scope('meta_condition_net', reuse=tf.AUTO_REUSE):
            nn_hidden = domain_emb
            for i, out_dim in enumerate(self.config.MCN_HIDDEN_LAYERS):
                nn_hidden = tf.layers.dense(
                    nn_hidden,
                    out_dim,
                    activation='relu',
                    use_bias=True,
                    name=f'mlp_{i}',
                    kernel_initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001)
                )
            out = tf.layers.dense(
                nn_hidden,
                num_expert_layers * 2,
                activation=None,
                use_bias=True,
                name=f'mlp_{i+1}',
                kernel_initializer=tf.random_uniform_initializer(minval=-0.001, maxval=0.001)
            )
            return out  # [B, 2*L]


    def _mlp_module_cbn(self, nn_input, delta_beta, delta_gamma, hidden_layers, act_func='relu', 
                        dropout_prob=0.1, batch_norm=False, inf_dropout=False, training=True, 
                        reuse=True, name_scope='mlp_module'):
        with tf.variable_scope(name_scope, reuse=reuse):
            hidden_output = nn_input
            for i, layer in enumerate(hidden_layers):
                hidden_output = tf.layers.dense(hidden_output, layer, activation=act_func,
                                                use_bias=True, name=f'mlp_{i}', reuse=reuse,
                                                kernel_initializer=tf.random_uniform_initializer(minval=-0.001, 
                                                    maxval=0.001))
                print(f'========= dense layer : {hidden_output.name} ==========')
                if batch_norm:
                    in_shape = hidden_output.get_shape().as_list()
                    if self.input_gate_method == 'trm' and len(in_shape) == 3:
                        # [B, num_tasks, D] -> [B, num_tasksxD]
                        hidden_output = tf.reshape(hidden_output, [-1, in_shape[1]*in_shape[2]])
                        hidden_output = ConditionalBatchNormalization(
                            in_shape[1]*in_shape[2],
                            scope=f"conditional_bn_{i}"
                        ).forward(
                            hidden_output,
                            delta_beta=delta_beta[:, i],
                            delta_gamma=delta_gamma[:, i],
                            training=training
                        )
                        hidden_output = tf.reshape(hidden_output, [-1, in_shape[1], in_shape[2]])
                    else:
                        hidden_output = ConditionalBatchNormalization(
                            in_shape[-1],
                            scope=f"conditional_bn_{i}"
                        ).forward(
                            hidden_output,
                            delta_beta=delta_beta[:, i],
                            delta_gamma=delta_gamma[:, i],
                            training=training
                        )

                if training or inf_dropout:
                    # dropout at inference time only if inf_dropout is True
                    print(f"""apply dropout in *{'training' if training else 'inference'}* time,
                          inf_dropout: {inf_dropout}""")
                    hidden_output = tf.nn.dropout(hidden_output, rate=dropout_prob)
                    
            return hidden_output


    def cal_gate_output(self, nn_input, num_tasks, gate_act_func, batch_norm, training):
        gate_outputs = []
        if self.weight_method == 'dr' or self.weight_method == 'drvb':
            num_task_gates = 2 * num_tasks - 1
        else:
            num_task_gates = num_tasks
        for i in range(num_task_gates):
            if self.input_gate_method == 'trm':
                gate_in = nn_input[:, i, :]
            elif self.enable_task_emb:
                enhance_task_emb = self.task_feat_generator(nn_input)
                if training:
                    self.task_cl_loss = self.cal_task_cl_loss_v2(enhance_task_emb, self.lbl_values)
                task_embs = tf.expand_dims(self.task_embs[i], axis=0)
                bs = tf.shape(nn_input)[0]
                task_embs = tf.repeat(task_embs, bs, axis=0)
                if self.stop_grad_global_task_emb:
                    task_embs = tf.stop_gradient(task_embs)
                gate_in = tf.concat([task_embs, enhance_task_emb], axis=-1)
            else:
                gate_in = nn_input

            gate_output = self.gate_forward(
                scope=f"gate_{i}_net",
                nn_input=gate_in,
                act_func=gate_act_func,
                batch_norm=batch_norm,
                training=training
            ) # [B, N_E]
            gate_outputs.append(gate_output) 
        gate_outputs = tf.stack(gate_outputs, axis=1) # [B, N_task, N_E]
        return gate_outputs


    def loss_part(self, final_layer_y, ptmzr_argv):
        super(MMOE_CrossScenario, self).loss_part(final_layer_y, ptmzr_argv)
        if (self.enable_domain_gate) and (self.config.ENHANCE_DOMAIN_FEAT):
            self.loss = self.loss + self.config.AUX_LAMBDA * self.domain_cl_loss
        if self.enable_task_emb:
            self.loss = self.loss + self.config.AUX_LAMBDA_TASK * self.task_cl_loss
