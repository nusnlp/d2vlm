# ------------------------------------------------------------------------
# D2VLM
# Copyright (c) 2025 Wenzheng Zeng. Licensed under the BSD-3-Clause license.
# ------------------------------------------------------------------------
# Modified from E.T. Chat (https://github.com/PolyU-ChenLab/ETBench)
# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.
# ------------------------------------------------------------------------

import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen2Model

from d2vlm.model.d2vlm_arch import D2VLMMetaForCausalLM, D2VLMMetaModel


class ETChatQwen2Config(Qwen2Config):
    model_type = 'etchat_qwen2'


class ETChatQwen2Model(D2VLMMetaModel, Qwen2Model):
    config_class = ETChatQwen2Config


class ETChatQwen2ForCausalLM(D2VLMMetaForCausalLM, Qwen2ForCausalLM):
    config_class = ETChatQwen2Config

    def __init__(self, config):
        super().__init__(config)
        self.model = ETChatQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


AutoConfig.register('etchat_qwen2', ETChatQwen2Config)
AutoModelForCausalLM.register(ETChatQwen2Config, ETChatQwen2ForCausalLM)
