import os
import cv2
from PIL import Image
from aot_tracker import _palette
import numpy as np
import torch
from scipy.ndimage import binary_dilation
import gc
from app import (
    init_SegTracker,
    SegTracker_add_first_frame,
    get_meta_from_video,
    tracking_objects
)


def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)


def draw_mask(img, mask, alpha=0.5, id_countour=False):
    img_mask = np.copy(img)
    if id_countour:
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        for id in obj_ids:
            if id <= 255:
                color = _palette[id * 3:id * 3 + 3]
            else:
                color = [0, 0, 0]
            foreground = img * (1 - alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)
            img_mask[binary_mask] = foreground[binary_mask]

            contours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
            img_mask[contours, :] = 0
    else:
        binary_mask = (mask != 0)
        contours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
        foreground = img * (1 - alpha) + colorize_mask(mask) * alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[contours, :] = 0

    return img_mask.astype(img.dtype)


def gd_detect(Seg_Tracker, origin_frame, grounding_caption='people', box_threshold=0.25,
              text_threshold=0.25, aot_model='r50_deaotl', long_term_mem=9999, max_len_long_term=9999, sam_gap=100,
              max_obj_num=255, points_per_side=16):
    if Seg_Tracker is None:
        Seg_Tracker, _, _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num,
                                               points_per_side, origin_frame)

    print("Detect")
    predicted_mask, annotated_frame = Seg_Tracker.detect_and_seg(origin_frame, grounding_caption, box_threshold,
                                                                  text_threshold)

    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)

    masked_frame = draw_mask(annotated_frame, predicted_mask)

    return Seg_Tracker, masked_frame, origin_frame


video_name = 'cctv_cut'
io_args = {
    'input_video': f'./assets/{video_name}.mp4',
    'output_mask_dir': f'./assets/{video_name}_masks',  # save pred masks
    'output_video': f'./assets/{video_name}_seg.mp4',  # mask+frame visualization, mp4 or avi, else the same as input video
    'output_gif': f'./assets/{video_name}_seg.gif',  # mask visualization
}

segtracker_args = {
    'sam_gap': 5,  # the interval to run sam to segment new objects
    'min_area': 200,  # minimal mask area to add a new mask as a new object
    'max_obj_num': 255,  # maximal object number to track in a video
    'min_new_obj_iou': 0.8,  # the area of a new object in the background should > 80%
    'box_threshold': 0.25,
    'text_threshold': 0.25,
    'aot_model': 'r50_deaotl',
    'long_term_mem': 9999,
    'max_len_long_term': 9999,
    'points_per_side': 16,
}

# Source video to segment
cap = cv2.VideoCapture(io_args['input_video'])
fps = cap.get(cv2.CAP_PROP_FPS)
# Output masks
output_dir = io_args['output_mask_dir']
os.makedirs(output_dir, exist_ok=True)

torch.cuda.empty_cache()
gc.collect()

# Reading Data
input_first_frame, origin_frame, drawing_board, grounding_caption = get_meta_from_video(io_args['input_video'])
grounding_caption = 'people'

box_threshold = segtracker_args['box_threshold']
text_threshold = segtracker_args['text_threshold']
aot_model = segtracker_args['aot_model']
long_term_mem = segtracker_args['long_term_mem']
max_len_long_term = segtracker_args['max_len_long_term']
sam_gap = segtracker_args['sam_gap']
max_obj_num = segtracker_args['max_obj_num']
points_per_side = segtracker_args['points_per_side']

Seg_Tracker, _, _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num,
                                       points_per_side, origin_frame)
Seg_Tracker, input_first_frame, _ = gd_detect(Seg_Tracker, origin_frame, grounding_caption,
                                              box_threshold, text_threshold,
                                              aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num,
                                              points_per_side)
output_video, output_mask, objs_list = tracking_objects(Seg_Tracker, io_args['input_video'], None, 0)

print('Output Video:', output_video)
print('Object List:', objs_list)
