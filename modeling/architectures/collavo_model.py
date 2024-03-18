# CoLLaVO
import torch
from torch import nn
from torch.nn import functional as F

from .build import register_model
from ..utils import configurable
from collavo.utils.utils import *

# CoLLaVO
from collavo.load_collavo import prepare_collavo

class CoLLaVO(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        cfg,
        collavo_model, # CoLLaVO
        collavo_processor, # CoLLaVO
        seg_model,
        seg_processor
    ):
        super().__init__()
        self.cfg = cfg
        self.collavo_model = collavo_model # CoLLaVO
        self.collavo_processor = collavo_processor # CoLLaVO
        self.seg_model = seg_model # CoLLaVO
        self.seg_processor = seg_processor # CoLLaVO

    @classmethod
    def from_config(cls, cfg):

        # CoLLaVO
        if cfg['LLM']['LOAD_LLM']:
            collavo_model, collavo_processor, seg_model, seg_processor = prepare_collavo(bits=cfg['LLM']['BITS'],
                                                                                        grad_ckpt=cfg['LLM']['GRAD_CKPT'],
                                                                                        lora=cfg['LLM']['LORA'],
                                                                                        dtype=cfg['LLM']['DTYPE'])
        else:
            collavo_model, collavo_processor = None, None

        return {
            "cfg": cfg,
            "collavo_model": collavo_model, # CoLLaVO
            "collavo_processor": collavo_processor, # CoLLaVO
            "seg_model": seg_model, # CoLLaVO
            "seg_processor": seg_processor, # CoLLaVO
            }

@register_model
def get_collavo_model(cfg, **kwargs):
    return CoLLaVO(cfg)