import os
import sqlite3
import argparse
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import schedule
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
class Segmenter_Tracker:
    def __int__(self):
        pass

    def colorize_mask(self, pred_mask):
        save_mask = Image.fromarray(pred_mask.astype(np.uint8))
        save_mask = save_mask.convert(mode='P')
        save_mask.putpalette(_palette)
        save_mask = save_mask.convert(mode='RGB')
        return np.array(save_mask)

    def draw_mask(self, img, mask, alpha=0.5, id_countour=True):
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
            foreground = img * (1 - alpha) + self.colorize_mask(mask) * alpha
            img_mask[binary_mask] = foreground[binary_mask]
            img_mask[contours, :] = 0

        return img_mask.astype(img.dtype)

    def gd_detect(self, Seg_Tracker, origin_frame, grounding_caption='people', box_threshold=0.25,
                  text_threshold=0.25, aot_model='r50_deaotl', long_term_mem=9999, max_len_long_term=9999, sam_gap=100,
                  max_obj_num=255, points_per_side=16):
        if Seg_Tracker is None:
            Seg_Tracker, _, _, _ = init_SegTracker(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num,
                                                   points_per_side, origin_frame)

        predicted_mask, annotated_frame = Seg_Tracker.detect_and_seg(origin_frame, grounding_caption, box_threshold,
                                                                     text_threshold)

        Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)

        masked_frame = self.draw_mask(annotated_frame, predicted_mask, id_countour=True)

        return Seg_Tracker, masked_frame, origin_frame


class VideoDownloader:
    def __init__(self,
                 data_file_path,
                 gdrive_folder = 'Tracking',
                 grounding_caption = 'people',
                 local_folder = 'gdrive',
                 database = 'SegTracker.db'):
        self.data_file_path = os.path.join(local_folder, data_file_path)
        self.drive = self.authenticate()
        self.segmenter_obj = Segmenter_Tracker()
        self.video_directory = os.path.join(local_folder, 'downloaded_videos')
        self.gdrive_folder = gdrive_folder
        self.grounding_caption = grounding_caption
        self.database = database


        self.segtracker_args = {
            'sam_gap': 1000,  # the interval to run sam to segment new objects
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

        torch.cuda.empty_cache()
        gc.collect()

        self.box_threshold = self.segtracker_args['box_threshold']
        self.text_threshold = self.segtracker_args['text_threshold']
        self.aot_model = self.segtracker_args['aot_model']
        self.long_term_mem = self.segtracker_args['long_term_mem']
        self.max_len_long_term = self.segtracker_args['max_len_long_term']
        self.sam_gap = self.segtracker_args['sam_gap']
        self.max_obj_num = self.segtracker_args['max_obj_num']
        self.points_per_side = self.segtracker_args['points_per_side']



    def authenticate(self):
        # Authenticate using client_secrets.json
        gauth = GoogleAuth()
        # gauth.LocalWebserverAuth()
        scope = ['https://www.googleapis.com/auth/drive']
        from oauth2client.service_account import ServiceAccountCredentials
        gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name('genuine-park-393209-c6843b80e8d2.json', scope)

        drive = GoogleDrive(gauth)

        return drive

    def download_recent_videos(self):
        folder_id = ''

        # 2) Retrieve the folder id - start searching from root
        file_list = self.drive.ListFile().GetList()
        for file in file_list:
            if file['title'] == self.gdrive_folder:
                folder_id = file['id']
                break

        # 3) Build string dynamically (need to use escape characters to support single quote syntax)
        query_str = "\'" + folder_id + "\'" + " in parents and trashed=false"

        # 4) Starting iterating over files
        recent_videos = self.drive.ListFile({'q': query_str}).GetList()
        new_videos = []
        if recent_videos:
            for video in recent_videos:
                file_path = os.path.join(self.video_directory, video['title'])
                if self.check_text_file(video['title']):
                    video.GetContentFile(file_path)
                    new_videos.append(video['title'])
                    print(f"Downloaded video: {video['title']}")
        return new_videos
    def check_text_file(self, file):
        if not os.path.exists(self.data_file_path):
            # Creates a new file
            with open(self.data_file_path, 'w') as fp:
                pass

        with open(self.data_file_path, "r+") as file1:
            # Reading from a file
            files = file1.read()
        if file in files:
            print(f'Already Done File ! {file}')
            return False
        else:
            with open(self.data_file_path, 'a') as file1:
                file1.write(file + '\n')
            return True

    def run_process(self, video_file):
        print(f"Running process for video: {video_file}")
        io_args = {
            'input_video': f'{self.video_directory}/{video_file}.mp4',
            'output_mask_dir': f'{self.video_directory}/{video_file}_masks',  # save pred masks
            'output_video': f'{self.video_directory}/{video_file}_seg.mp4',
            'output_gif': f'{self.video_directory}/{video_file}_seg.gif',  # mask visualization
        }
        input_first_frame, origin_frame, drawing_board, grounding_caption = get_meta_from_video(io_args['input_video'])
        Seg_Tracker, _, _, _ = init_SegTracker(self.aot_model, self.long_term_mem, self.max_len_long_term, self.sam_gap,
                                               self.max_obj_num,
                                               self.points_per_side, origin_frame)
        Seg_Tracker, input_first_frame, _ = self.segmenter_obj.gd_detect(Seg_Tracker, origin_frame, self.grounding_caption,
                                                           self.box_threshold, self.text_threshold,
                                                           self.aot_model, self.long_term_mem, self.max_len_long_term,
                                                           self.sam_gap, self.max_obj_num,
                                                           self.points_per_side)
        output_video, output_mask, objs_list = tracking_objects(Seg_Tracker, io_args['input_video'], None, 0)
        return output_video, objs_list[-1]


    def check_for_new_videos(self, new_videos):
        for new_video in new_videos:
            video_name = new_video.split('.')[0]
            output_video, objs_list = self.run_process(video_name)

            print('Output Video:', output_video)
            print('Object List:', objs_list)

            # Connect to the SQLite database
            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()

            # Create the videos table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS videos (
                    video_name TEXT,
                    metadata TEXT
                )
            ''')
            # Insert video information into the database
            cursor.execute('''
                INSERT INTO videos (video_name, metadata) VALUES (?, ?)
            ''', (video_name, str(objs_list)))
            # Commit the changes to the database
            conn.commit()
            # Close the database connection
            conn.close()

    def job(self):
        print("Scanning for recent videos...")
        new_videos = self.download_recent_videos()
        print('New Videos ', new_videos)
        self.check_for_new_videos(new_videos)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--file", default = 'already_looked_files.txt',  help="Text file which saves previously done files")
parser.add_argument("--gdrive", default = 'Tracking',  help="Google Drive Folder which contains new and old videos")
parser.add_argument("--caption", default = 'people',  help="Caption for Grounding Dino to detect onyl specific subjects")
parser.add_argument("--directory", default = './gdrive',  help="Local Folder to download Google Drive new files")
parser.add_argument("--database", default = 'SegTracking.db',  help="Local Database")
# Read arguments from command line
args = parser.parse_args()

video_downloader = VideoDownloader(args.file,
                                   args.gdrive,
                                   args.caption,
                                   args.directory,
                                   args.database)

# Schedule the job to run every 1 hour
schedule.every(1).minutes.do(video_downloader.job)

# Connect to the SQLite database


while True:
    schedule.run_pending()
    # time.sleep(1)
