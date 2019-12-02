def load_data(path):

    data = []

    frames = glob(path + '/*.jpg')
    
    for f in tqdm(frames):
        img = imageio.imread(f)
        data.append(cv2.resize(img, (100, 100), interpolation = cv2.INTER_AREA))
                         
    return np.array(data)