# ------------------------------------------------------------------------
# D2VLM
# Copyright (c) 2025 Wenzheng Zeng. Licensed under the BSD-3-Clause license.
# ------------------------------------------------------------------------
# Modified from E.T. Chat (https://github.com/PolyU-ChenLab/ETBench)
# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.
# ------------------------------------------------------------------------

import torch.nn as nn
import importlib
from transformers import AutoConfig, AutoModelForCausalLM, Phi3Config, Phi3ForCausalLM, Phi3Model

def get_etchat_meta_classes(arch_module):
    module = importlib.import_module(f'd2vlm.model.{arch_module}')
    return module.D2VLMMetaForCausalLM, module.D2VLMMetaModel

class ETChatPhi3Config(Phi3Config):
    model_type = 'etchat_phi3'
    def __init__(self, arch_module='etchat_arch', **kwargs):
        self.arch_module = arch_module
        super().__init__(**kwargs)

class ETChatPhi3Model(Phi3Model):
    config_class = ETChatPhi3Config
    
    def __init__(self, config):
        _, D2VLMMetaModel = get_etchat_meta_classes(config.arch_module)
        ETChatPhi3Model.__bases__ = (D2VLMMetaModel, Phi3Model)
        super().__init__(config)

class ETChatPhi3ForCausalLM(Phi3ForCausalLM):
    config_class = ETChatPhi3Config

    def __init__(self, config):
        D2VLMMetaForCausalLM, _ = get_etchat_meta_classes(config.arch_module)
        ETChatPhi3ForCausalLM.__bases__ = (D2VLMMetaForCausalLM, Phi3ForCausalLM)
        super().__init__(config)
        self.model = ETChatPhi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

AutoConfig.register('etchat_phi3', ETChatPhi3Config)
AutoModelForCausalLM.register(ETChatPhi3Config, ETChatPhi3ForCausalLM)