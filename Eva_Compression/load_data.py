def load_data(path):

    data = []

    frames = glob(path + '/*.jpg')
    
    for f in tqdm(frames):
        img = imageio.imread(f)
        data.append(cv2.resize(img, (100, 100), interpolation = cv2.INTER_AREA))
                         
    data_load = np.array(data)
    data_load= (data_load - 255)
    data_load = data_load/255

    batch_size = 8

    tensor_x = torch.stack([torch.Tensor(i) for i in data_load])

    train_dataset = utils.TensorDataset(tensor_x,tensor_x)
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader