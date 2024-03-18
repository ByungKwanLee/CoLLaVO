"""
CoLLaVO-7B

Simple Six Steps
"""

# [1] Loading Image
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor
image_path = "figures/crayon_image.jpg"
image = Resize(size=(490, 490), antialias=False)(pil_to_tensor(Image.open(image_path)))

# [2] Instruction Prompt
prompt = "Describe this image in detail."

# [3] Loading CoLLaVO
from collavo.load_collavo import prepare_collavo
collavo_model, collavo_processor, seg_model, seg_processor = prepare_collavo(collavo_path='BK-Lee/CoLLaVO-7B', bits=4, dtype='fp16')

# [4] Pre-processing for CoLLaVO
collavo_inputs = collavo_model.demo_process(image=image, 
                                    prompt=prompt, 
                                    processor=collavo_processor,
                                    seg_model=seg_model,
                                    seg_processor=seg_processor,
                                    device='cuda:0')

# [5] Generate
import torch
with torch.inference_mode():
    generate_ids = collavo_model.generate(**collavo_inputs, do_sample=True, temperature=0.9, top_p=0.95, max_new_tokens=256, use_cache=True)

# [6] Decoding
answer = collavo_processor.batch_decode(generate_ids, skip_special_tokens=True)[0].split('[U')[0]
print(answer)