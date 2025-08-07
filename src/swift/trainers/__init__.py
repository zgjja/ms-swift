# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import (EvaluationStrategy, FSDPOption, HPSearchBackend, HubStrategy, IntervalStrategy,
                                        SchedulerType)
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule
from . import callback

try:
    # https://github.com/huggingface/transformers/pull/25702
    from transformers.trainer_utils import ShardedDDPOption
except ImportError:
    ShardedDDPOption = None

if TYPE_CHECKING:
    from .arguments import (GRPOArgumentsMixin, RLHFArgumentsMixin, Seq2SeqTrainingArguments, TrainingArguments,
                            VllmArguments)
    from .mixin import SwiftMixin
    from .rlhf_arguments import CPOConfig, DPOConfig, GKDConfig, KTOConfig, ORPOConfig, PPOConfig, RewardConfig
    from .rlhf_trainer import (CPOTrainer, DPOTrainer, GKDTrainer, GRPOTrainer, KTOTrainer, ORPOTrainer, PPOTrainer,
                               RewardTrainer, RLHFTrainerMixin)
    from .trainer_factory import TrainerFactory
    from .trainers import EmbeddingTrainer, RerankerTrainer, Seq2SeqTrainer, Trainer

else:
    _extra_objects = {k: v for k, v in globals().items() if not k.startswith('_')}
    _import_structure = {
        'arguments':
        ['Seq2SeqTrainingArguments', 'TrainingArguments', 'RLHFArgumentsMixin', 'VllmArguments', 'GRPOArgumentsMixin'],
        'rlhf_arguments':
        ['DPOConfig', 'CPOConfig', 'KTOConfig', 'ORPOConfig', 'PPOConfig', 'RewardConfig', 'GRPOConfig', 'GKDConfig'],
        'rlhf_trainer': [
            'CPOTrainer', 'DPOTrainer', 'KTOTrainer', 'ORPOTrainer', 'RLHFTrainerMixin', 'PPOTrainer', 'RewardTrainer',
            'GRPOTrainer', 'GKDTrainer'
        ],
        'trainer_factory': ['TrainerFactory'],
        'trainers': ['Seq2SeqTrainer', 'Trainer', 'EmbeddingTrainer', 'RerankerTrainer'],
        'mixin': ['SwiftMixin'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects=_extra_objects,
    )
