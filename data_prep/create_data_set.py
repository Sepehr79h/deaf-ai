import json
from preprocess import *
import numpy as np
import dask.array as da

# def concat_mm(a_mmap,b,a_count):
#     c = np.memmap(a_mmap, dtype=np.float32, mode='r+', shape=(a_count,100,144,256), order='F')
#     c[a_count-1,:,:,:] = b
#     return c

def clip_and_label(file_loc="data_prep/WLASL_v0.3.json",output="/data/",raw_vids="raw_videos_mp4/"):
    f = open(file_loc)
    manf = json.load(f)

    pos_dataset = da.zeros((1,100,144,256), chunks=(1,100,144,256))
    neg_dataset = da.zeros((1,100,144,256), chunks=(1,100,144,256))

    pos_count = 1
    neg_count = 1

    for sl_class in manf:
        for vid in sl_class['instances']:
            if pos_count % 200 == 0:
                da.to_npy_stack('prep_data/sign_npy/',pos_dataset)
                da.to_npy_stack('prep_data/no_sign_npy/',neg_dataset)
            
            with open('prep_data/info.txt','w') as f:
                f.write("Number of Samples positive: {0}, negative {1}".format(pos_count,neg_count))

            frames = video_to_frames(raw_vids+vid['video_id']+'.mp4',(256,144))
            if len(frames) == 0:
                continue
            print("Labeling video {0}".format(vid['video_id']))
            sign_frames = frames[vid['frame_start']-1:vid['frame_end']]
            pos_np = da.array(sign_frames)
            pos_count +=1
            #normalzing size and saving to overarching list
            empty_entry = da.zeros((1,100,144,256))
            empty_entry[0,0:pos_np.shape[0]] = pos_np.copy()[0:min(100,len(pos_np))]
            pos_dataset = da.concatenate((pos_dataset,empty_entry.copy()),axis=0)
            #pos_dataset.flush()

            if vid['frame_start'] != 1 or vid['frame_end'] != -1:
                if vid['frame_start'] != 1:
                    no_sign_frames = frames[0:vid['frame_start']-1]
                elif vid['frame_end'] != -1:
                    no_sign_frames = frames[vid['frame_end']-1:-1]
                neg_np = np.array(no_sign_frames)
                neg_count += 1
                #normalzing size and saving to overarching list
                empty_entry = da.zeros((1,100,144,256))
                empty_entry[0,0:neg_np.shape[0]] = neg_np.copy()[0:min(100,len(neg_np))]
                neg_dataset = da.concatenate((neg_dataset,empty_entry.copy()),axis=0)
                #neg_dataset.flush()

    return pos_dataset,neg_dataset

if __name__=="__main__":
    signing,not_signing = clip_and_label()
    