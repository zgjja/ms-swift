# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .deploy import SwiftDeploy, deploy_main, run_deploy
    from .infer import SwiftInfer, infer_main
    from .infer_engine import (AdapterRequest, BaseInferEngine, InferClient, InferEngine, LmdeployEngine, PtEngine,
                               SglangEngine, VllmEngine, prepare_generation_config)
    from .protocol import Function, RequestConfig
    from .rollout import rollout_main
    from .utils import prepare_model_template
else:
    _import_structure = {
        'rollout': ['rollout_main'],
        'infer': ['infer_main', 'SwiftInfer'],
        'deploy': ['deploy_main', 'SwiftDeploy', 'run_deploy'],
        'protocol': ['RequestConfig', 'Function'],
        'utils': ['prepare_model_template'],
        'infer_engine': [
            'InferEngine', 'VllmEngine', 'LmdeployEngine', 'SglangEngine', 'PtEngine', 'InferClient',
            'prepare_generation_config', 'AdapterRequest', 'BaseInferEngine'
        ],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
