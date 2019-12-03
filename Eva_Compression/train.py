#This function serves to train CAE model on batches of shuffled and resized video frames
def train(model):

    model.train() #setting model to train 

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        torch.cuda.empty_cache() #clearing cache of previous batch data

        images, targets = batch[0], batch[0]
        images, targets = images.cuda(), targets.cuda()

        output = model(images)#Running with encoding flag set to False

        loss = criterion(output, targets) 
        loss = torch.sqrt(loss)
        #loss is now an RMSE values
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print("Epoch : ",epoch, " - Loss : ",loss)

    return loss
