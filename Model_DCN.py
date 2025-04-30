from __future__ import print_function

import sys
import os
import json
import numpy as np
import tensorflow as tf

from train.layer.CoreLayer import EmbeddingLayer, DNNLayer, LinearLayer, PredictionLayer
from train.layer.InteractionLayer import CrossLayer


class Model_DCN:

    def __init__(self, config):
        self.config = config

        self.id_hldr = None
        self.wt_hldr = None
        self.lbl_hldr = None
        self.global_step = None
        self.train_preds = None
        self.loss = None
        self.ptmzr = None
        self.eval_id_hldr = None
        self.eval_wt_hldr = None
        self.eval_preds = None

        self.run()

    def run(self):

        embedding_layer = EmbeddingLayer(self.config)
        cross_layer = CrossLayer(self.config, embedding_layer.output_tensor)
        dnn_layer = DNNLayer(self.config, embedding_layer.output_tensor).output_tensor
        linear_layer = LinearLayer(self.config, [cross_layer.output_tensor, dnn_layer.output_tensor])
        prediction_layer = PredictionLayer(self.config, linear_layer.output_tensor)

        self.set_out_node(embedding_layer, prediction_layer)

    def set_out_node(self, embedding_layer, prediction_layer):
        self.id_hldr = embedding_layer.id_hldr
        self.wt_hldr = embedding_layer.wt_hldr
        self.global_step = prediction_layer.global_step
        self.lbl_hldr = prediction_layer.lbl_hldr
        self.train_preds = prediction_layer.train_preds
        self.loss = prediction_layer.loss
        self.ptmzr = prediction_layer.ptmzr
        self.eval_id_hldr = embedding_layer.wt_hldr
        self.eval_wt_hldr = embedding_layer.id_hldr
        self.eval_preds = prediction_layer.train_preds
