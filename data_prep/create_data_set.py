import json
from preprocess import *
import numpy as np
from video_downloader import download_youtube
import moviepy.editor as mp
import random
import math 
import pandas as pd
import more_itertools as mit
from tqdm import tqdm
import dask.array as da

# def concat_mm(a_mmap,b,a_count):
#     c = np.memmap(a_mmap, dtype=np.float32, mode='r+', shape=(a_count,100,144,256), order='F')
#     c[a_count-1,:,:,:] = b
#     return c

# def clip_and_label(file_loc="data_prep/WLASL_v0.3.json",output="/data/",raw_vids="raw_videos_mp4/"):
#     f = open(file_loc)
#     manf = json.load(f)

#     pos_dataset = da.zeros((1,100,144,256), chunks=(1,100,144,256))
#     neg_dataset = da.zeros((1,100,144,256), chunks=(1,100,144,256))

#     pos_count = 1
#     neg_count = 1

#     for sl_class in manf:
#         for vid in sl_class['instances']:
#             if pos_count % 200 == 0:
#                 da.to_npy_stack('prep_data/sign_npy/',pos_dataset)
#                 da.to_npy_stack('prep_data/no_sign_npy/',neg_dataset)
            
#             with open('prep_data/info.txt','w') as f:
#                 f.write("Number of Samples positive: {0}, negative {1}".format(pos_count,neg_count))

#             frames = video_to_frames(raw_vids+vid['video_id']+'.mp4',(256,144))
#             if len(frames) == 0:
#                 continue
#             print("Labeling video {0}".format(vid['video_id']))
#             sign_frames = frames[vid['frame_start']-1:vid['frame_end']]
#             pos_np = da.array(sign_frames)
#             pos_count +=1
#             #normalzing size and saving to overarching list
#             empty_entry = da.zeros((1,100,144,256))
#             empty_entry[0,0:pos_np.shape[0]] = pos_np.copy()[0:min(100,len(pos_np))]
#             pos_dataset = da.concatenate((pos_dataset,empty_entry.copy()),axis=0)
#             #pos_dataset.flush()

#             if vid['frame_start'] != 1 or vid['frame_end'] != -1:
#                 if vid['frame_start'] != 1:
#                     no_sign_frames = frames[0:vid['frame_start']-1]
#                 elif vid['frame_end'] != -1:
#                     no_sign_frames = frames[vid['frame_end']-1:-1]
#                 neg_np = np.array(no_sign_frames)
#                 neg_count += 1
#                 #normalzing size and saving to overarching list
#                 empty_entry = da.zeros((1,100,144,256))
#                 empty_entry[0,0:neg_np.shape[0]] = neg_np.copy()[0:min(100,len(neg_np))]
#                 neg_dataset = da.concatenate((neg_dataset,empty_entry.copy()),axis=0)
#                 #neg_dataset.flush()

#     return pos_dataset,neg_dataset

# def clip_and_label_no_sign(raw_vids="no_sign_vids/"):
#     neg_dataset = da.zeros((1,100,144,256), chunks=(1,100,144,256))
#     for count in range(51):      
#         frames = video_to_frames(raw_vids+str(count)+'.mp4',(256,144))
#         if len(frames) == 0:
#             continue
#         print("Labeling video {0}".format(count))
#         neg_np = np.array(frames)
#         empty_entry = da.zeros((1,100,144,256))
#         empty_entry[0,0:neg_np.shape[0]] = neg_np.copy()[0:min(100,len(neg_np))]
#         neg_dataset = da.concatenate((neg_dataset,empty_entry.copy()),axis=0)
#     da.to_npy_stack('prep_data/no_sign_npy/',neg_dataset)

# def get_yt_vids(urls='./data_prep/yt_vid_list.txt'):
#     with open(urls,'r') as r:
#         for url in r.readlines():
#             frames = download_youtube(url.strip(),'./raw_yt_vids',0)

# def cut_vid_yt(vdir='./raw_yt_vids'):
#     for vfile in os.listdir(vdir):
#         f = os.path.join(vdir,vfile)
#         video = VideoFileClip(f)

#         durr = video.duration

#         for i in range(int(durr//11)):
#             start = random.randint(0,int(durr)-11)
#             ffmpeg_extract_subclip(f, start, start+6, targetname="no_sign_vids/"+vfile+str(i)+".mp4")

def create_siw_dataset(output='siw_data/siw_raw_data',vid_list='data_prep/sign_in_wild/vid_list.txt',limit=1,download=False):
    if download:
        download_siw_videos(vid_list,limit)
        #resize_videos(output)
    frames = label_process_siw(output)
    gen_siw_array(output,frames)

def resize_videos(vid_dir):
    clips = os.listdir(vid_dir)
    print("RESIZING VIDEOS TO NORMAL RESOLUTION")
    for clip in tqdm(clips):
        try:
            vid = mp.VideoFileClip(vid_dir+'/'+clip)
            vid_resized = vid.resize(height=144,width=256) # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
            vid_resized.write_videofile(vid_dir+'/'+clip)
        except:
            continue

def download_siw_videos(vid_list,limit):
    existing_dls = os.listdir('siw_data/siw_raw_data')
    with open(vid_list,'r') as r:
        urls = r.readlines()
        print("DOWNLOADING VIDEOS")
        for url in tqdm(urls[:math.floor(len(urls)*limit)]):
            if url.strip().split("=")[1]+'.mp4' in existing_dls:
                print("{0} - already downloaded".format(url.strip()))
                continue
            try:
                download_youtube(url.strip(),'siw_data/siw_raw_data',0)
            except:
                continue

def label_process_siw(data):
    vid_frames = {}
    ordering_6=[x for x in range(0,100000,6)]
    ordering_5=[x for x in range(0,100000,5)]
    cols = ['ids','frame','label']
    gt = pd.read_csv('data_prep/sign_in_wild/groundtruth.txt',names=cols,delimiter=' ',header=None)
    print("LABELING VIDEOS")
    gt['ids']=gt['ids']+'.mp4'

    avg_len_no_sign = []
    avg_len_sign = []
    def average_len(l):
        return sum(map(len, l))/float(len(l)+.00001)

    for vid_id in tqdm(gt['ids'].unique()):
        if (vid_id not in os.listdir('siw_data/siw_raw_data')):
            continue
        processed = True
        signing_frames = []
        nosign_frames = []
        signing = gt[(gt['label']=='S') & (gt['ids']==vid_id)]
        no_sign = gt[(gt['label']=='P') & (gt['ids']==vid_id)]

        

        #check for 6 frame sampling
        try:
            if signing.shape[0] != 0:
                for group in mit.consecutive_groups(signing['frame'],ordering=ordering_6.index):
                    signing_frames.append(list(group))
            if no_sign.shape[0] != 0:
                for group in mit.consecutive_groups(no_sign['frame'],ordering=ordering_6.index):
                    nosign_frames.append(list(group))
        except:
            processed = False
        if not processed:
            #check for 5 frame sampling
            try:
                if signing.shape[0] != 0:
                    for group in mit.consecutive_groups(signing['frame'],ordering=ordering_5.index):
                        signing_frames.append(list(group))
                if no_sign.shape[0] != 0:
                    for group in mit.consecutive_groups(no_sign['frame'],ordering=ordering_5.index):
                        nosign_frames.append(list(group))
            except:
                continue
        
        avg_len_no_sign += nosign_frames
        avg_len_sign += signing_frames
        vid_frames[vid_id] = [signing_frames,nosign_frames]

    return vid_frames
            
def gen_siw_array(data,frames):
    print("GATHERING VIDEO FRAMES")
    for vid in tqdm(list(frames.keys())[int(len(frames.keys())/2):]):
        curr_vid = video_to_frames(data+'/'+vid)
        curr_signf = frames[vid][0]
        curr_nsignf = frames[vid][1]
        count = 0
        existing_dls = os.listdir('siw_data/sign_clips')
        for chunk in curr_signf:
            count+=1
            if vid.split('.')[0]+'-'+str(count)+'.avi' in existing_dls:
                # print("{0} - already downloaded".format(url.strip()))
                continue
            if chunk[-1]-chunk[0] >= 100:
                upper_i = [i for i in range(len(chunk)) if chunk[i]-chunk[0] >= 94][0]
                curr_signf.append(chunk[upper_i:])
                chunk = chunk[:upper_i]
            #print("NUMBER OF FRAMES FOR SAMPLE: "+str(chunk[-1]-chunk[0]))
            video_frames = curr_vid[chunk[0]:chunk[-1]]
            if len(video_frames) == 0:
              continue
            convert_frames_to_video(np.array(video_frames),'siw_data/sign_clips/{0}-{1}.avi'.format(vid.split('.')[0],count),(len(video_frames[0][0]),len(video_frames[0])))
            

        for chunk in curr_nsignf:
            count+=1
            if chunk[-1]-chunk[0] >= 100:
                upper_i = [i for i in range(len(chunk)) if chunk[i]-chunk[0] >= 100][0]
                curr_nsignf.append(chunk[upper_i:])
                chunk = chunk[:upper_i]
            # print("NUMBER OF FRAMES FOR SAMPLE: "+str(chunk[-1]-chunk[0]))
            # video_frames = da.array(video_to_frames(data+'/'+vid,(256,144))[chunk[0]:chunk[-1]+1])
            # empty_entry = da.zeros((1,100,144,256))
            # empty_entry[0,0:video_frames.shape[0]] = video_frames.copy()[0:min(100,len(video_frames))]
            # nsign_data = da.concatenate((nsign_data,empty_entry.copy()),axis=0)
            
            #video_frames = curr_vid[chunk[0]:chunk[-1]]
            continue
            if len(video_frames) == 0:
              continue
            
            #convert_frames_to_video(np.array(video_frames),'siw_data/nosign_clips/{0}-{1}.avi'.format(vid.split('.')[0],count),(len(video_frames[0][0]),len(video_frames[0])))
        
        if len(frames) == 0:
            continue
    # da.to_npy_stack('siw_data/sign_npy/',sign_data)
    # da.to_npy_stack('siw_data/no_sign_npy/',nsign_data)

def siw_to_npy(data_location='./siw_data'):
    sign_dataset = da.zeros((1,100,240,320), chunks=(10,100,240,320))
    nosign_dataset = da.zeros((1,100,240,320), chunks=(10,100,240,320))
    sign_dir=data_location+'/sign_clips'
    nosign_dir = data_location+'/nosign_clips'
    sign_samples = os.listdir(sign_dir)
    nosign_samples = os.listdir(nosign_dir)

    print("CONVERTING SIGN CLIPS TO NPY")
    for s in tqdm(sign_samples):
        curr_frames = video_to_frames(sign_dir+'/'+s,(320,240))
        empty_entry = da.zeros((1,100,240,320))
        empty_entry[0,0:len(curr_frames)] = curr_frames.copy()[0:min(100,len(curr_frames))]
        sign_dataset = da.concatenate((sign_dataset,empty_entry.copy()),axis=0)
    da.to_npy_stack(data_location+'/npy_data/sign_clips',sign_dataset)

    print("CONVERTING NO=SIGN CLIPS TO NPY")
    for s in tqdm(nosign_samples):
        curr_frames = video_to_frames(nosign_dir+'/'+s,(320,240))
        empty_entry = da.zeros((1,100,240,320))
        empty_entry[0,0:len(curr_frames)] = curr_frames.copy()[0:min(100,len(curr_frames))]
        nosign_dataset = da.concatenate((nosign_dataset,empty_entry.copy()),axis=0)
    da.to_npy_stack(data_location+'/npy_data/nosign_clips',nosign_dataset)

if __name__=="__main__":
    #signing,not_signing = clip_and_label()
    #clip_and_label_no_sign()
    #get_yt_vids()
    #cut_vid_yt()
    #create_siw_dataset(limit=.5)
    #siw_to_npy()
    label_process_siw('siw_data/siw_raw_data')
    