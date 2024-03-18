import torch
import numpy as np
from detectron2.structures import BitMasks
from utils.constants import COCO_PANOPTIC_CLASSES

def make_system_prompt(processor, device, ignore_index, img_length=1225):
    # system prompt
    system_prompt = make_human_string("AI assistant should give helpful and detailed answers to user after fully understanding an image.",
                                    "<image>")

    length = processor(system_prompt, return_tensors='pt').input_ids[0].shape[0]
    collavo_label = torch.tensor([ignore_index]*(length+img_length-1)).to(device)
    im_mask = torch.zeros_like(collavo_label)
    im_mask[-img_length:]=1

    return system_prompt, collavo_label, im_mask

def demo_make_and_add_prompt_and_im_mask(collavo_prompt, im_mask, prompt, processor, device):
    
    # indent
    prompt = " USER: " + prompt + " ASSISTANT:"

    # input_ids and 
    label_ids = processor(prompt, return_tensors='pt', add_special_tokens=False).input_ids[0]
    
    # Concat previous prompt + current prompt
    collavo_prompt += prompt
    im_mask = torch.tensor(im_mask.tolist() + torch.zeros_like(label_ids).tolist()).to(device)
    
    return collavo_prompt, im_mask
    
def make_and_add_prompt_and_im_mask(collavo_prompt, im_mask, prompt, processor, device):
    
    # indent
    prompt = " [UNUSED_TOKEN_146]user\n" + prompt + "[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"

    # input_ids and 
    label_ids = processor(prompt, return_tensors='pt', add_special_tokens=False).input_ids[0]
    
    # Concat previous prompt + current prompt
    collavo_prompt += prompt
    im_mask = torch.tensor(im_mask.tolist() + torch.zeros_like(label_ids).tolist()).to(device)
    
    return collavo_prompt, im_mask

def make_human_string(*args):
    out = ''
    for ind, arg in enumerate(args):
        out += arg
        if len(args)-1 != ind: out += ' '
    return out

def box_and_class_parser(decoded_text):
    start_box_index = find(decoded_text, '[')
    end_box_index = find(decoded_text, ']')

    start_class_index = find(decoded_text, '(')
    end_class_index = find(decoded_text, ')')
    
    if len(start_box_index) != len(end_box_index): return None, None, True
    if len(start_class_index) != len(end_class_index): return None, None, True
    if len(start_class_index) != len(start_box_index): return None, None, True

    box_list = []
    class_list = []
    for sb, eb, sc, ec in zip(start_box_index, end_box_index, start_class_index, end_class_index):
        box_list.append(eval(decoded_text[sb: eb+1]))
        class_list.append(decoded_text[sc+1: ec][decoded_text[sc+1: ec].find(' ')+1:])
        if len(box_list[-1]) != 4: box_list.pop(-1); class_list.pop(-1)
    box_tensor = torch.tensor(box_list)
    return box_tensor, class_list, False

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def list2string(_list):
    out = ''
    for i, x in enumerate(_list):
        out+=str(x)
        if i!=len(_list)-1: out+=', '
    out += ''
    return out

def box2string(box):
    out = '['
    for i, x in enumerate(box):
        out+=f"{round(x.item(), 2):.2f}"
        if i!=len(box)-1: out+=', '
    out += ']'
    return out

def boxes2string(boxes):
    out = ''
    for i, x in enumerate(boxes):
        out+=box2string(x)
        if i!=len(boxes)-1: out+=', '
    out += ''
    return out

def classescolors2string(nice_seg_info_list):
    classes = [nice_seg_info['class'] for nice_seg_info in nice_seg_info_list]
    colors = [nice_seg_info['color'] for nice_seg_info in nice_seg_info_list]

    count = {}
    out = ''
    for i, (x, y) in enumerate(zip(classes, colors)):
        if x in count.keys():
            count[x]+=1
        else:
            count.update({x: 1})
        out+=f"({count[x]} {x}) {y}"
        if i!=len(classes)-1: out+=', '
    return out


def classesboxes2string(nice_seg_info_list, class_name='all'):
    if class_name == 'all':
        classes = [nice_seg_info['class'] for nice_seg_info in nice_seg_info_list]
        boxes = [nice_seg_info['box'] for nice_seg_info in nice_seg_info_list]
    else:
        classes = [nice_seg_info['class'] for nice_seg_info in nice_seg_info_list if nice_seg_info['class']==class_name]
        boxes = [nice_seg_info['box'] for nice_seg_info in nice_seg_info_list if nice_seg_info['class']==class_name]

    count = {}
    out = ''
    for i, (x, y) in enumerate(zip(classes, boxes)):
        if x in count.keys():
            count[x]+=1
        else:
            count.update({x: 1})
        out+=f"(#{count[x]} {x}) {box2string(y)}"
        if i!=len(classes)-1: out+=', '
    return out

def classes2string(nice_seg_info_list):
    classes = [nice_seg_info['class'] for nice_seg_info in nice_seg_info_list]
    count = {}
    out = ''
    for i, x in enumerate(classes):
        if x in count.keys():
            count[x]+=1
        else:
            count.update({x: 1})
        out+=f"#{count[x]} {x}"
        if i!=len(classes)-1: out+=', '
    return out


def create_pascal_label_colormap(num=1+133):
    def bit_get(val, idx):
        return (val >> idx) & 1
    colormap = np.zeros((num, 3), dtype=int)
    ind = np.arange(num, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3
    return colormap / 255

def make_seg_prompt(seg_ind, seg_info):
    seg_ind[torch.where(seg_ind<=0)] = 0 # Filtering unknown and background
    seg_prompt_ind = torch.zeros_like(seg_ind)
    seg_num_ind = torch.zeros_like(seg_ind)

    count = {}
    for s_ind in seg_ind.unique():
        if s_ind == 0: continue
        for s_info in seg_info:
            if s_info['id'] == s_ind:
                label_id = s_info['label_id']
                if label_id in count.keys():
                    count[label_id] += 1
                else:
                    count.update({label_id: 1})
                break
        seg_prompt_ind[torch.where(seg_ind == s_ind)] = label_id+1
        seg_num_ind[torch.where(seg_ind == s_ind)] = count[label_id]

    # Generating Boxes
    boxes = BitMasks(torch.stack([seg_ind == i+1 for i in range(seg_ind.max()) if (seg_ind == i+1).sum() != 0])).get_bounding_boxes()
    boxes.scale(1/35, 1/35)

    # Panoptic Index to seg info including box
    nice_seg_info_list = []
    for i, e in enumerate(seg_info):
        nice_seg_info_list.append(
        {'id': e['id'],
            'class': COCO_PANOPTIC_CLASSES[e['label_id']].replace('-merged','').replace('-other','').replace('-stuff',''),
            'box': boxes.tensor[i]
            })

    return seg_prompt_ind, seg_num_ind, nice_seg_info_list

