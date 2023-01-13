import torch.optim as optim
import torch.nn as nn
from . import models
from torch.utils.data.dataloader import DataLoader
import traintracker
from traintracker import TrainTracker, TrackerMod
import torch
from typing import Union


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

    model_layers = model.hidden_shape + [model.output_size]

    hyperparameters = {"batch_size": train_size[1], "optimizer": optimizer, "model_layers": model_layers}
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

    return train_losses, valid_losses


def lstm_test(model: models.SalesLstm, valid_loader, loss_function, train_tracker: TrainTracker, device='cpu'):
    model.eval()
    train_tracker.valid()
    with torch.no_grad():
        hidden = None
        for sales_in, store_data_in, targets in valid_loader:
            sales_in, store_data_in, targets = sales_in.to(device), store_data_in.to(device), targets.to(device)

            predicted_output, hidden = model(sales_in, store_data_in, hidden)

            loss = loss_function(predicted_output, targets)
            avg_test_loss = train_tracker.step(loss.item())
    return avg_test_loss


def lstm_train(model: models.SalesLstm, train_loader: DataLoader, valid_loader: DataLoader, n_epochs, device='cpu',
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

    if 'grad_clip' not in kwargs:
        grad_clip = 5
    else:
        grad_clip = kwargs['grad_clip']

    train_size = len(train_loader), train_loader.batch_size

    valid_size = len(valid_loader), valid_loader.batch_size

    lstm_layers = model.lstm_architecture
    nn_layers = model.nn_architecture
    fc_output_layer = model.fcn_architecture

    seq_len = train_loader.dataset.seq_length

    hyperparameters = {"batch_size": train_size[1], "optimizer": optimizer, "lstm_layers": lstm_layers,
                       "nn_layers": nn_layers, "fc_output_layer": fc_output_layer, "seq_len": seq_len,
                       "grad_clip": grad_clip}

    if 'notes' in kwargs:
        hyperparameters['notes'] = kwargs['notes']

    train_tracker = TrainTracker(model=model, tracker_mod=TrackerMod.TRAIN_TEST, train_data_size=train_size,
                                 test_data_size=valid_size, train_data_dir=kwargs['train_data_dir'],
                                 hyperparameters=hyperparameters, weights_dir=kwargs['weights_dir'],
                                 last_weights=last_weights)

    train_losses, valid_losses = [], []
    print("Testing before training")
    test_tracker = traintracker.TrainTracker(model, tracker_mod=TrackerMod.TEST_ONLY, test_data_size=valid_size)
    test_loss = lstm_test(model, valid_loader, loss_function, test_tracker, device=device)
    print(f"Test Loss :{round(test_loss, 3)}")

    for e in range(n_epochs):
        model.train()
        train_tracker.train()
        hidden = None
        for sales_in, store_data, targets in train_loader:
            sales_in, store_data, targets = sales_in.to(device), store_data.to(device), targets.to(device)

            predicted_output, hidden = model(sales_in, store_data, hidden)

            loss = loss_function(predicted_output, targets)
            loss.backward()

            train_tracker.step(loss.item())
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            optimizer.zero_grad()
            hidden = tuple(h.data for h in hidden)

        lstm_test(model, valid_loader, loss_function, train_tracker, device)

        train_loss, valid_loss = train_tracker.end_epoch()

        train_losses.append(train_loss)

        valid_losses.append(valid_loss)

    return train_losses, valid_losses
