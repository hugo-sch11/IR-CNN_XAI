import src.data.a_cifar10 as ds
import src.model.cnn_cifar10 as cnn

import time
import torch

def train_and_save_model(train_times: int) -> None:
    starting_time = time.time()
    print("Beginning Training")
    ConvNet = cnn.ConvNet1()
    for epoch in range(train_times): # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(ds.trainloader, 0):
            # get the inputs, data is a list of [inputs, labels]
            inputs, labels = data

            # set the gradient to zero
            ConvNet.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = ConvNet(inputs)
            loss: torch.Tensor = ConvNet.criterion(outputs, labels)
            loss.backward()
            ConvNet.optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
    ending_time = time.time()
    print("Finished Training")
    print(f"Trained for : {ending_time - starting_time:.2f}seconds")
    torch.save(ConvNet.state_dict(), "./cifar_net.pth")