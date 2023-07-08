import os
import cv2
from SegTracker import SegTracker
from model_args import aot_args, sam_args, segtracker_args
from PIL import Image
from aot_tracker import _palette
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc
from app import init_SegTracker, SegTracker_add_first_frame, get_meta_from_video, tracking_objects

def save_prediction(pred_mask, output_dir, file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir, file_name))


def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)


def draw_mask(img, mask, alpha=0.5, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id * 3:id * 3 + 3]
            else:
                color = [0, 0, 0]
            foreground = img * (1 - alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask != 0)
        countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
        foreground = img * (1 - alpha) + colorize_mask(mask) * alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours, :] = 0

    return img_mask.astype(img.dtype)


def create_dir(dir_path):
    # if os.path.isdir(dir_path):
    #     os.system(f"rm -r {dir_path}")

    # os.makedirs(dir_path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def gd_detect(Seg_Tracker, origin_frame, grounding_caption = 'people', box_threshold= 0.25,
              text_threshold= 0.25, aot_model='r50_deaotl', long_term_mem= 9999, max_len_long_term=9999, sam_gap=100,
              max_obj_num=255, points_per_side=16):
    if Seg_Tracker is None:
        Seg_Tracker, _ , _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)

    print("Detect")
    predicted_mask, annotated_frame= Seg_Tracker.detect_and_seg(origin_frame, grounding_caption, box_threshold, text_threshold)

    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)


    masked_frame = draw_mask(annotated_frame, predicted_mask)

    return Seg_Tracker, masked_frame, origin_frame


def video_type_input_tracking(SegTracker, input_video, io_args, video_name, frame_num=0):
    pred_list = []
    masked_pred_list = []

    # source video to segment
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_num > 0:
        output_mask_name = sorted([img_name for img_name in os.listdir(io_args['output_mask_dir'])])
        output_masked_frame_name = sorted([img_name for img_name in os.listdir(io_args['output_masked_frame_dir'])])

        for i in range(0, frame_num):
            cap.read()
            pred_list.append(
                np.array(Image.open(os.path.join(io_args['output_mask_dir'], output_mask_name[i])).convert('P')))
            masked_pred_list.append(
                cv2.imread(os.path.join(io_args['output_masked_frame_dir'], output_masked_frame_name[i])))

    # create dir to save predicted mask and masked frame
    if frame_num == 0:
        if os.path.isdir(io_args['output_mask_dir']):
            os.system(f"rm -r {io_args['output_mask_dir']}")
        if os.path.isdir(io_args['output_masked_frame_dir']):
            os.system(f"rm -r {io_args['output_masked_frame_dir']}")
    output_mask_dir = io_args['output_mask_dir']
    create_dir(io_args['output_mask_dir'])
    create_dir(io_args['output_masked_frame_dir'])

    torch.cuda.empty_cache()
    gc.collect()
    sam_gap = SegTracker.sam_gap
    frame_idx = 0

    with torch.cuda.amp.autocast():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_idx == 0:
                pred_mask = SegTracker.first_frame_mask
                torch.cuda.empty_cache()
                gc.collect()
            elif (frame_idx % sam_gap) == 0:
                seg_mask = SegTracker.seg(frame)
                torch.cuda.empty_cache()
                gc.collect()
                track_mask = SegTracker.track(frame)
                # find new objects, and update tracker with new objects
                new_obj_mask = SegTracker.find_new_objs(track_mask, seg_mask)
                save_prediction(new_obj_mask, output_mask_dir, str(frame_idx + frame_num).zfill(5) + '_new.png')
                pred_mask = track_mask + new_obj_mask
                # segtracker.restart_tracker()
                SegTracker.add_reference(frame, pred_mask)
            else:
                pred_mask = SegTracker.track(frame, update_memory=True)
            torch.cuda.empty_cache()
            gc.collect()

            save_prediction(pred_mask, output_mask_dir, str(frame_idx + frame_num).zfill(5) + '.png')
            pred_list.append(pred_mask)

            print("processed frame {}, obj_num {}".format(frame_idx + frame_num, SegTracker.get_obj_num()), end='\r')
            frame_idx += 1
        cap.release()
        print('\nfinished')

    ##################
    # Visualization
    ##################

    # draw pred mask on frame and save as a video
    cap = cv2.VideoCapture(input_video)
    # if frame_num > 0:
    #     for i in range(0, frame_num):
    #         cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # if input_video[-3:]=='mp4':
    #     fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
    # elif input_video[-3:] == 'avi':
    #     fourcc =  cv2.VideoWriter_fourcc(*"MJPG")
    #     # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # else:
    #     fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred_mask = pred_list[frame_idx]
        masked_frame = draw_mask(frame, pred_mask)
        cv2.imwrite(f"{io_args['output_masked_frame_dir']}/{str(frame_idx).zfill(5)}.png", masked_frame[:, :, ::-1])

        masked_pred_list.append(masked_frame)
        masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
        out.write(masked_frame)
        print('frame {} writed'.format(frame_idx), end='\r')
        frame_idx += 1
    out.release()
    cap.release()
    print("\n{} saved".format(io_args['output_video']))
    print('\nfinished')

    # save colorized masks as a gif
    imageio.mimsave(io_args['output_gif'], masked_pred_list, fps=fps)
    print("{} saved".format(io_args['output_gif']))

    # zip predicted mask
    os.system(f"zip -r {io_args['tracking_result_dir']}/{video_name}_pred_mask.zip {io_args['output_mask_dir']}")

    # manually release memory (after cuda out of memory)
    del SegTracker
    torch.cuda.empty_cache()
    gc.collect()

    return io_args['output_video'], f"{io_args['tracking_result_dir']}/{video_name}_pred_mask.zip"


video_name = 'cctv_cut'
io_args = {
    'input_video': f'./assets/{video_name}.mp4',
    'output_mask_dir': f'./assets/{video_name}_masks', # save pred masks
    'output_video': f'./assets/{video_name}_seg.mp4', # mask+frame vizualization, mp4 or avi, else the same as input video
    'output_gif': f'./assets/{video_name}_seg.gif', # mask visualization
}

# For every sam_gap frames, we use SAM to find new objects and add them for tracking
# larger sam_gap is faster but may not spot new objects in time
segtracker_args = {
    'sam_gap': 5,  # the interval to run sam to segment new objects
    'min_area': 200,  # minimal mask area to add a new mask as a new object
    'max_obj_num': 255,  # maximal object number to track in a video
    'min_new_obj_iou': 0.8,  # the area of a new object in the background should > 80%
}

# source video to segment
cap = cv2.VideoCapture(io_args['input_video'])
fps = cap.get(cv2.CAP_PROP_FPS)
# output masks
output_dir = io_args['output_mask_dir']
# draw pred mask on frame and save as a video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if io_args['input_video'][-3:]=='mp4':
    fourcc =  cv2.VideoWriter_fourcc(*"mp4v")
elif io_args['input_video'][-3:] == 'avi':
    fourcc =  cv2.VideoWriter_fourcc(*"MJPG")
    # fourcc = cv2.VideoWriter_fourcc(*"XVID")
else:
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
out = cv2.VideoWriter(io_args['output_video'], fourcc, fps, (width, height))

frame_idx = 0

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
pred_list = []
masked_pred_list = []

torch.cuda.empty_cache()
gc.collect()
sam_gap = segtracker_args['sam_gap']
frame_idx = 0


### Reading Data
input_first_frame, origin_frame, drawing_board, grounding_caption = get_meta_from_video(io_args['input_video'])

grounding_caption = 'people'
box_threshold= 0.25
text_threshold= 0.25
aot_model='r50_deaotl'
long_term_mem= 9999
max_len_long_term=9999
sam_gap=100
max_obj_num=255
points_per_side=16


Seg_Tracker, _, _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame)
Seg_Tracker, input_first_frame, _ = gd_detect(Seg_Tracker, origin_frame, grounding_caption,
                                          box_threshold, text_threshold,
                                          aot_model, long_term_mem, max_len_long_term, sam_gap,max_obj_num, points_per_side)
output_video, output_mask = tracking_objects(Seg_Tracker,
                io_args['input_video'],
                None, 0)
#
#
# segtracker = SegTracker(segtracker_args, sam_args, aot_args)
# segtracker.restart_tracker()
#
# with torch.cuda.amp.autocast():
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         if frame_idx == 0:
#             pred_mask = segtracker.seg(frame)
#             torch.cuda.empty_cache()
#             gc.collect()
#             segtracker.add_reference(frame, pred_mask)
#         elif (frame_idx % sam_gap) == 0:
#             seg_mask = segtracker.seg(frame)
#             torch.cuda.empty_cache()
#             gc.collect()
#             track_mask = segtracker.track(frame)
#             # find new objects, and update tracker with new objects
#             new_obj_mask = segtracker.find_new_objs(track_mask, seg_mask)
#             save_prediction(new_obj_mask, output_dir, str(frame_idx) + '_new.png')
#             pred_mask = track_mask + new_obj_mask
#             # segtracker.restart_tracker()
#             segtracker.add_reference(frame, pred_mask)
#         else:
#             pred_mask = segtracker.track(frame, update_memory=True)
#         masked_frame = draw_mask(frame, pred_mask)
#         masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
#         out.write(masked_frame)
#         print('frame {} writed'.format(frame_idx), end='\r')
#         frame_idx += 1
#
#         torch.cuda.empty_cache()
#         gc.collect()
#         save_prediction(pred_mask, output_dir, str(frame_idx) + '.png')
#         # masked_frame = draw_mask(frame,pred_mask)
#         # masked_pred_list.append(masked_frame)
#         # plt.imshow(masked_frame)
#         # plt.show()
#
#         pred_list.append(pred_mask)
#
#         print("processed frame {}, obj_num {}".format(frame_idx, segtracker.get_obj_num()), end='\r')
#         frame_idx += 1
#     cap.release()
#     out.release()
#
#     print("\n{} saved".format(io_args['output_video']))
#     print('\nfinished')
#
