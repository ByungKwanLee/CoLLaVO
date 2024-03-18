import torch

from .arch_collavo import CoLLaVOModel
from .arch.tokenization_internlm2 import CoLLaVOTokenizer

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    output_lora_module_names = list(lora_module_names)
    output_lora_module_names.remove("Plora_A")
    output_lora_module_names.remove("Plora_B")
    output_lora_module_names.remove("Plora_seg_A")
    output_lora_module_names.remove("Plora_seg_B")
    return output_lora_module_names

def prepare_collavo(bits, dtype):

    # Mask2Former-Panoptic
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    # load Mask2Former fine-tuned on COCO panoptic segmentation
    seg_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
    seg_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
    # "huggingface/hub/models--facebook--mask2former-swin-large-coco-panoptic/snapshots/85b535928a783691eaf27467a573b26d543336ea"

    # CoLLaVO
    bnb_model_from_pretrained_args = {}
    if bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            torch_dtype=torch.bfloat16 if dtype=='bf16' else torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2", # Flash attention, but it is not possible to extract attention output once it is enabled.
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=bits == 4,
                load_in_8bit=bits == 8,
                llm_int8_skip_modules=["vision_tower", "vision_proj", "vision_proj_for_seg", "Plora_main", "Plora_seg_A", "Plora_seg_B", "output"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16 if dtype=='bf16' else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        ))

    # CoLLaVOModel Loading
    collavo_model = CoLLaVOModel.from_pretrained("BK-Lee/CoLLaVO-7B", **bnb_model_from_pretrained_args)
    collavo_model.model.config.use_cache = False

    # bfloat16/float16 conversion 
    for param in collavo_model.parameters():
        if 'float32' in str(param.dtype).lower():
            param.data = param.data.to(torch.bfloat16 if dtype=='bf16' else torch.float16)
    
    # Post-Processing for <image> Token
    collavo_processor = CoLLaVOTokenizer.from_pretrained("BK-Lee/CoLLaVO-7B", padding_side='left')
    collavo_processor.add_tokens("<image>", special_tokens=True)
    collavo_model.resize_token_embeddings(len(collavo_processor))
    collavo_model.config.image_token_index = collavo_processor("<image>", add_special_tokens=False, return_tensors='pt').input_ids.item()
    collavo_model.config.ignore_index = -100
    collavo_model.config.pad_token_id = -100
    return collavo_model, collavo_processor, seg_model, seg_processor

# for name, param in collavo_model.named_parameters(): print(f"{name}: {param.dtype} {param.requires_grad}")