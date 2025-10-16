# ------------------------------------------------------------------------
# D2VLM
# Copyright (c) 2025 Wenzheng Zeng. Licensed under the BSD-3-Clause license.
# ------------------------------------------------------------------------
# Modified from E.T. Chat (https://github.com/PolyU-ChenLab/ETBench)
# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.
# ------------------------------------------------------------------------

import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, Gemma2Config, Gemma2ForCausalLM, Gemma2Model

from d2vlm.model.d2vlm_arch import D2VLMMetaForCausalLM, D2VLMMetaModel


class ETChatGemma2Config(Gemma2Config):
    model_type = 'etchat_gemma2'


class ETChatGemma2Model(D2VLMMetaModel, Gemma2Model):
    config_class = ETChatGemma2Config


class ETChatGemma2ForCausalLM(D2VLMMetaForCausalLM, Gemma2ForCausalLM):
    config_class = ETChatGemma2Config

    def __init__(self, config):
        super().__init__(config)
        self.model = ETChatGemma2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


AutoConfig.register('etchat_gemma2', ETChatGemma2Config)
AutoModelForCausalLM.register(ETChatGemma2Config, ETChatGemma2ForCausalLM)
