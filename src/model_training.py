import torch.optim as optim
import torch.nn as nn
from . import models
from torch.utils.data.dataloader import DataLoader
import traintracker
from traintracker import TrainTracker, TrackerMod
import torch


def nn_model_valid(model: models.SalesNN, valid_loader, loss_function, train_tracker: TrainTracker, device='cpu'):
    model.eval()
    train_tracker.valid()
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            predicted_output = model(inputs)

            loss = loss_function(predicted_output, targets)
            avg_test_loss = train_tracker.step(loss.item())
    return avg_test_loss


def nn_model_train(model: models.SalesNN, train_loader: DataLoader, valid_loader: DataLoader, n_epochs, device='cpu',
                   last_weights=False,
                   **kwargs):
    # ini optimizer and loss function

    if 'optimizer' not in kwargs:
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    else:
        optimizer = kwargs['optimizer']
    if 'loss_function' not in kwargs:
        loss_function = nn.MSELoss()
    else:
        loss_function = kwargs['loss_function']

    train_size = len(train_loader), train_loader.batch_size

    valid_size = len(valid_loader), valid_loader.batch_size

    hyperparameters = {"batch_size": train_size[1], "optimizer": optimizer}
    if 'notes' in kwargs:
        hyperparameters['notes'] = kwargs['notes']

    train_tracker = TrainTracker(model=model, tracker_mod=TrackerMod.TRAIN_TEST, train_data_size=train_size,
                                 test_data_size=valid_size, train_data_dir=kwargs['train_data_dir'],
                                 hyperparameters=hyperparameters, weights_dir=kwargs['weights_dir'],
                                 last_weights=last_weights)
    train_losses = []
    valid_losses = []

    for e in range(n_epochs):
        model.train()
        train_tracker.train()

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)

            predicted_output = model(inputs)

            loss = loss_function(predicted_output, targets)
            loss.backward()

            train_tracker.step(loss.item())
            optimizer.step()

        nn_model_valid(model, valid_loader, loss_function, train_tracker, device)

        train_loss, valid_loss = train_tracker.end_epoch()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
    return train_losses,valid_losses
