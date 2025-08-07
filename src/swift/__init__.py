# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from .utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .trainers import (EvaluationStrategy, FSDPOption, HPSearchBackend, HubStrategy, IntervalStrategy,
                           SchedulerType, Seq2SeqTrainer, Seq2SeqTrainingArguments, ShardedDDPOption, Trainer,
                           TrainingArguments)
    from .tuners import (SWIFT_MAPPING, AdaLoraConfig, Adapter, AdapterConfig, AdapterModule, LoftQConfig, LoHaConfig,
                         LoKrConfig, LongLoRA, LongLoRAConfig, LongLoRAModelType, LoRA, LoRAConfig, LoraConfig,
                         OFTConfig, PeftConfig, PeftModel, PeftModelForCausalLM, PeftModelForSeq2SeqLM,
                         PeftModelForSequenceClassification, PeftModelForTokenClassification, PrefixTuningConfig,
                         Prompt, PromptConfig, PromptEncoderConfig, PromptLearningConfig, PromptModule,
                         PromptTuningConfig, ResTuningConfig, SCETuning, SCETuningConfig, SideConfig, Swift,
                         SwiftConfig, SwiftModel, SwiftOutput, SwiftTuners, get_peft_config, get_peft_model,
                         get_peft_model_state_dict)
    from .utils import get_logger
    from .version import __release_datetime__, __version__
else:
    _import_structure = {
        'version': ['__release_datetime__', '__version__'],
        'tuners': [
            'Adapter', 'AdapterConfig', 'AdapterModule', 'SwiftModel', 'LoRA', 'LoRAConfig', 'SWIFT_MAPPING',
            'LoraConfig', 'AdaLoraConfig', 'LoftQConfig', 'LoHaConfig', 'LoKrConfig', 'OFTConfig', 'PeftConfig',
            'ResTuningConfig', 'SideConfig', 'PeftModel', 'PeftModelForCausalLM', 'PeftModelForSeq2SeqLM',
            'PeftModelForSequenceClassification', 'PeftModelForTokenClassification', 'PrefixTuningConfig',
            'PromptEncoderConfig', 'PromptLearningConfig', 'PromptTuningConfig', 'get_peft_config', 'get_peft_model',
            'get_peft_model_state_dict', 'Prompt', 'PromptConfig', 'PromptModule', 'SwiftConfig', 'SwiftOutput',
            'Swift', 'SwiftTuners', 'LongLoRAConfig', 'LongLoRA', 'LongLoRAModelType', 'SCETuning', 'SCETuningConfig'
        ],
        'trainers': [
            'EvaluationStrategy',
            'FSDPOption',
            'HPSearchBackend',
            'HubStrategy',
            'IntervalStrategy',
            'SchedulerType',
            'ShardedDDPOption',
            'TrainingArguments',
            'Seq2SeqTrainingArguments',
            'Trainer',
            'Seq2SeqTrainer',
        ],
        'utils': ['get_logger']
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
