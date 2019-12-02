data = []

def Write_Video(w=None,h=None,path='',Detrac = False):

    if Detrac:
        c = 0
        width = 960
        height = 540
        fourcc = cv2.VideoWriter_fourcc(*'MP42')
        for i in os.listdir('DETRAC-Images/'):
            video = cv2.VideoWriter('./video_0.avi',fourcc,30, (width,height))
            for j in range(1, len(os.listdir('DETRAC-Images/' + i + '/'))+1):
                #print(j)
                j = str(j)
                jj= 5-len(j)
                k = "img" + jj*"0" +j +".jpg"
                img = imageio.imread('DETRAC-Images/' + i + '/' + k + '/')
                #print(img.shape)
                #plt.imshow(img)
                #plt.show()
                video.write(img)
        #         video.release()
        #         cv2.destroyAllWindows()
                
            video.release()
            cv2.destroyAllWindows()
            break
            
        video.release()
        cv2.destroyAllWindows()

    else:
        c = 0
        
        fourcc = cv2.VideoWriter_fourcc(*'MP42')
        frames = glob( path + '/*jpg') 

        width = frames[0].shape[1]
        height = frames[0].shape[0]
        video = cv2.VideoWriter('./video_0.avi',fourcc,30, (width,height))
        for img in frames:
            video.write(img)
    
        video.release()
        cv2.destroyAllWindows()
