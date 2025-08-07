# Copyright (c) Alibaba, Inc. and its affiliates.
import gradio as gr
from typing import Type

from swift.ui.llm_train.save import Save


class RLHFSave(Save):

    group = 'llm_rlhf'
