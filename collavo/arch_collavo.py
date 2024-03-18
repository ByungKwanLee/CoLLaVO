from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from dataclasses import dataclass
from transformers.utils.generic import ModelOutput
from torch import nn

from .utils.utils import *
from .arch.build_mlp import build_vision_projector, build_vision_tower
from .arch.modeling_internlm2 import InternLM2Model, InternLM2PreTrainedModel
from transformers.cache_utils import Cache

@dataclass
class CoLLaVOCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    seg_attentions: Optional[Tuple[torch.FloatTensor]] = None
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None

class CoLLaVOModel(InternLM2PreTrainedModel):
    _auto_class = 'AutoModelForCausalLM'

    _tied_weights_keys = ['output.weight']

    def __init__(self, config):
        super().__init__(config)
        self.model = InternLM2Model(config)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.max_length = config.max_length
        print(f'Set max length to {self.max_length}')
        
        # Initialize weights and apply final processing
        self.post_init()
        self.vit = build_vision_tower()
        self.vision_proj = build_vision_projector()

        # CoLLaVO
        self.vision_proj_for_seg = build_vision_projector() # 1024 -> 4096
        self.seg_prompt_embed = nn.Embedding(1+133, 1024) # num of coco-panoptic classes 133 + 1 (unknown)
        self.seg_num_embed = nn.Embedding(1+20, 1024) # num of maximum instances 20 + 1 (unknown)

        # image processing variable
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,-1,1,1) * 255
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,-1,1,1) * 255

    def image_processor(self, images):
        norm_images = (images - self.mean.to(images.device)) / self.std.to(images.device)
        return norm_images

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.config.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def eval_process(
        self,
        images,
        prompts,
        processor=None,
        seg_model=None,
        seg_processor=None,
        device=None):

        # Mask2former Inputs
        seg_inputs = seg_processor(images=[img for img in images], return_tensors="pt")

        # Mask2former Outputs
        with torch.inference_mode():
            seg_model.eval()
            seg_results = seg_processor.post_process_panoptic_segmentation(seg_model(**{k:v.to(device) for k, v in seg_inputs.items()}), 
                                                                           target_sizes=[(35, 35)]*len(images),
                                                                           threshold=0.5,
                                                                           mask_threshold=0.95,
                                                                           label_ids_to_fuse=())
        batched_collavo_prompt=[]
        batched_seg_prompt = []
        batched_seg_num = []
        batched_im_mask=[]
        for prompt, seg_result in zip(prompts, seg_results):

            # collavo prompt prefix
            collavo_prompt, _, im_mask = make_system_prompt(processor, device, self.config.ignore_index)

            # Panoptic Index and Class Index
            seg_ind = seg_result['segmentation'].clone().to(torch.int32)
            seg_info = seg_result['segments_info']

            # TRY-EXCEPT: Zero segmentation
            if (seg_ind>0).sum() == 0:                
                seg_prompt_ind = torch.zeros_like(seg_ind).to(device)
                seg_num_ind = torch.zeros_like(seg_ind).to(device)
                collavo_prompt, im_mask = make_and_add_prompt_and_im_mask(collavo_prompt=collavo_prompt, 
                                                                        im_mask=im_mask, 
                                                                        prompt=f"None of detailed object information for image. {prompt}", 
                                                                        processor=processor, 
                                                                        device=device)
            else:
                seg_prompt_ind, seg_num_ind, nice_seg_info_list = make_seg_prompt(seg_ind, seg_info)
                collavo_prompt, im_mask = make_and_add_prompt_and_im_mask(collavo_prompt=collavo_prompt, 
                                                                        im_mask=im_mask, 
                                                                        prompt=f"The image includes {classesboxes2string(nice_seg_info_list)}. {prompt}", 
                                                                        processor=processor, 
                                                                        device=device)

            # making batched collavo prompt
            batched_collavo_prompt.append(collavo_prompt)
            batched_seg_prompt.append(seg_prompt_ind.flatten())
            batched_seg_num.append(seg_num_ind.flatten())
            batched_im_mask.append(im_mask.flip(dims=[0])) # padding left
        
        '''For Final Outputs'''
        collavo_inputs = processor(batched_collavo_prompt, padding=True, return_tensors="pt")

        # [1] input_ids
        input_ids = collavo_inputs.input_ids.to(device)
        
        # [2] pixel values
        pixel_values = self.image_processor(images).to(device)

        # [3] seg values
        seg_values = (torch.stack(batched_seg_prompt).to(device), torch.stack(batched_seg_num).to(device))
        
        # [4] attention_mask
        attention_mask = collavo_inputs.attention_mask.to(device)
        
        # [5] im_mask
        im_mask = torch.nn.utils.rnn.pad_sequence(batched_im_mask, batch_first=True, padding_value=0).flip(dims=[1]).bool() # padding left

        return {"input_ids": input_ids, "pixel_values": pixel_values, "seg_values": seg_values, "attention_mask": attention_mask, "im_mask": im_mask}


    def demo_process(
        self,
        image,
        prompt,
        processor=None,
        seg_model=None,
        seg_processor=None,
        device=None):

        # RGB Dimension
        image = image[:3]

        # Mask2former Inputs
        seg_inputs = seg_processor(images=[image], return_tensors="pt")

        # Mask2former Outputs
        with torch.inference_mode():
            seg_model.eval()
            seg_results = seg_processor.post_process_panoptic_segmentation(seg_model(**{k:v.to(device) for k, v in seg_inputs.items()}), 
                                                                           target_sizes=[(35, 35)],
                                                                           threshold=0.5,
                                                                           mask_threshold=0.95,
                                                                           label_ids_to_fuse=())

        # collavo prompt prefix
        collavo_prompt, _, im_mask = make_system_prompt(processor, device, self.config.ignore_index)

        # Panoptic Index and Class Index
        seg_ind = seg_results[0]['segmentation'].clone().to(torch.int32)
        seg_info = seg_results[0]['segments_info']

        # TRY-EXCEPT: Zero segmentation
        if (seg_ind>0).sum() == 0:                
            seg_prompt_ind = torch.zeros_like(seg_ind).to(device)
            seg_num_ind = torch.zeros_like(seg_ind).to(device)
            collavo_prompt, im_mask = demo_make_and_add_prompt_and_im_mask(collavo_prompt=collavo_prompt, 
                                                                    im_mask=im_mask, 
                                                                    prompt=f"None of detailed object information for image. {prompt}", 
                                                                    processor=processor, 
                                                                    device=device)
        else:
            seg_prompt_ind, seg_num_ind, nice_seg_info_list = make_seg_prompt(seg_ind, seg_info)
            collavo_prompt, im_mask = demo_make_and_add_prompt_and_im_mask(collavo_prompt=collavo_prompt, 
                                                                    im_mask=im_mask, 
                                                                    prompt=f"The image includes {classesboxes2string(nice_seg_info_list)}. {prompt}", 
                                                                    processor=processor, 
                                                                    device=device)

        # making batched collavo prompt
        batched_collavo_prompt = [collavo_prompt]
        batched_seg_prompt = [seg_prompt_ind.flatten()]
        batched_seg_num = [seg_num_ind.flatten()]
        batched_im_mask = [im_mask.flip(dims=[0])] # padding left
        
        '''For Final Outputs'''
        collavo_inputs = processor(batched_collavo_prompt, padding=True, return_tensors="pt")

        # [1] input_ids
        input_ids = collavo_inputs.input_ids.to(device)
        
        # [2] pixel values
        pixel_values = self.image_processor(image.unsqueeze(0)).to(device)

        # [3] seg values
        seg_values = (torch.stack(batched_seg_prompt).to(device), torch.stack(batched_seg_num).to(device))
        
        # [4] attention_mask
        attention_mask = collavo_inputs.attention_mask.to(device)
        
        # [5] im_mask
        im_mask = torch.nn.utils.rnn.pad_sequence(batched_im_mask, batch_first=True, padding_value=0).flip(dims=[1]).bool() # padding left

        return {"input_ids": input_ids, "pixel_values": pixel_values, "seg_values": seg_values, "attention_mask": attention_mask, "im_mask": im_mask}

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            seg_values: Tuple = (None, None),
            attention_mask: Optional[torch.Tensor] = None,
            im_mask: torch.BoolTensor = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, CoLLaVOCausalLMOutputWithPast]:
            
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # CoLLaVO
        self.vit.vision_tower.eval()
        vision_attention = None
        seg_attention = None

        if inputs_embeds is None:

            # TODO: For batch generation
            input_ids[torch.where(input_ids==-100)]=2

            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.vit(pixel_values)
                seg_outputs = self.seg_prompt_embed.weight[seg_values[0].repeat_interleave(image_outputs.shape[0] // seg_values[0].shape[0], dim=0)] \
                    + self.seg_num_embed.weight[seg_values[1].repeat_interleave(image_outputs.shape[0] // seg_values[1].shape[0], dim=0)]

                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                image_features = self.vision_proj(image_outputs)
                seg_features = self.vision_proj_for_seg(seg_outputs)
                inputs_embeds, attention_mask, _, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )
                if labels is None:
                    labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)
            else:
                # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
                # generation with cache
                if past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                    # Retrieve the first layer to inspect the logits and mask out the hidden states
                    # that are set to 0
                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                    # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                    # Get the target length
                    target_seqlen = first_layer_past_key_value.shape[-1] + 1

                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], target_seqlen - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )

                    # Filter out only the tokens that can be un-attended, this can happen
                    # if one uses Llava + Fused modules where the cache on the
                    # first iteration is already big enough, or if one passes custom cache
                    valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                    new_batch_index = batch_index[valid_indices]
                    new_non_attended_tokens = non_attended_tokens[valid_indices]

                    # Zero-out the places where we don't need to attend
                    extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                    attention_mask = torch.cat((attention_mask, extended_attention_mask), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                    im_mask = torch.zeros(inputs_embeds.shape[:2]).bool().to(inputs_embeds.device)
                    seg_features = None

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            seg_mask=seg_features,
            im_mask=im_mask,
        )
        logits = self.output(outputs[0])

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # Try except handling for use_cache=True
        return CoLLaVOCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            vision_attentions=vision_attention,
            seg_attentions=seg_attention,
            language_attentions=outputs.attentions,
        )


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, seg_values=None, attention_mask=None, im_mask=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "seg_values": seg_values,
                "im_mask": im_mask,
            }
        )
        return model_inputs
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past