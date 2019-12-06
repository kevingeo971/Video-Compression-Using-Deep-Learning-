import numpy as np
import cv2
from tqdm import tqdm 

#We wish to save a compressed video that only contains represenative frames
def save_compressed_video(original_frames, path_to,index_list):
    print('\n Saving Compressed video .. ')
    original_frames = np.array(original_frames)
    frames = original_frames[index_list]
    
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    
    width = frames[0].shape[1]
    height = frames[0].shape[0]

    video = cv2.VideoWriter(path_to,fourcc,30, (width,height))

    print()
    for f in tqdm(frames):
        video.write(f)
    
    video.release()
    cv2.destroyAllWindows()

    return video 



    



