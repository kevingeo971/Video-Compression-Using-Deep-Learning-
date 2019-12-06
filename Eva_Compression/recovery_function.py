import numpy as np
import cv2

#TO DO add numpy slicing the list operations

def recovery_function(frame_array, metadata):
    path = metadata.original_path

    sorted_frame_array = sorted(frame_array)
    arg_array = frame_array.argsort()
    arg_array = arg_array.argsort()
    
    frames = []
    flat_frames = []
    clip_lengths = []

    for cf in sorted_frame_array:
        subframes = list(range(metadata.compressed_index[cf] , metadata.compressed_index[cf+1]))
        frames.append(subframes)
        flat_frames.extend(subframes)
        clip_lengths.append(len(subframes))
    
    cap = cv2.VideoCapture(path)   

    output_frames = []

    c = 0

    while(cap.isOpened()):
        frameId = int(cap.get(1)) #current frame number
        
        ret, frame = cap.read()
        if (ret != True):
            break

        if frameId == flat_frames[c]:
            output_frames.append(frame)
            if c < (len(flat_frames) -1):
                c+=1
            else:
                break

        cap.release()

    restruct_output_frames = []

    c = 0

    for l in clip_lengths:
        subframes = output_frames[c:c+l]
        c += l
        restruct_output_frames.append(subframes)

    restruct_output_frames = restruct_output_frames[arg_array]
    


        

