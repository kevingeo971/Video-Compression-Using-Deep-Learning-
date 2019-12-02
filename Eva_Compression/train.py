def train():

    model.train()

    c = 0

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        torch.cuda.empty_cache()
        #print(batch[0].shape)
        images, targets = batch[0], batch[0]
        images, targets = images.cuda(), targets.cuda()

        output = model(images)

        loss = criterion(output, targets)
        loss = torch.sqrt(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print("Epoch : ",epoch, " - Loss : ",loss)

    return loss
