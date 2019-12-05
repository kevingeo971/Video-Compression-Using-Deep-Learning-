#data = []
'''
This function takes in path as the argument. It write the video into "save_path" 
'''
import cv2
import os
import imageio
from glob import glob

def write_data(index_list,data_path='',save_path='',Detrac = False):

    if Detrac:
        c = 0
        width = 960
        height = 540
        fourcc = cv2.VideoWriter_fourcc(*'MP42')
        for i in os.listdir('DETRAC-Images/'):
            video = cv2.VideoWriter('./DETRAC_video.avi',fourcc,30, (width,height))
            for j in range(1, len(os.listdir('DETRAC-Images/' + i + '/'))+1):
                j = str(j)
                jj= 5-len(j)
                k = "img" + jj*"0" +j +".jpg"
                img = imageio.imread('DETRAC-Images/' + i + '/' + k + '/')
                video.write(img)
                
            video.release()
            cv2.destroyAllWindows()
            break
            
        video.release()
        cv2.destroyAllWindows()

    else:
        c = 0
        
        fourcc = cv2.VideoWriter_fourcc(*'MP42')
        frames = glob( data_path + '/*jpg') 

        width = frames[0].shape[1]
        height = frames[0].shape[0]
        video = cv2.VideoWriter(save_path,fourcc,30, (width,height))
        for img in frames:
            video.write(img)
    
        video.release()
        cv2.destroyAllWindows()

        
        