import torch
import torch.nn as nn

salesNN_parameters = {

}


class SalesNN(nn.Module):
    def __init__(self, input_size, hidden_shape, output_size=1, dropout_prop=0.5):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_shape = hidden_shape

        self.dropout = nn.Dropout(dropout_prop)
        self.model = nn.Sequential(nn.Linear(input_size, hidden_shape[0]), nn.ReLU(), nn.Dropout(dropout_prop))

        # hidden layers
        for i in range(len(hidden_shape) - 1):
            self.model.append(nn.Linear(hidden_shape[i], hidden_shape[i + 1]))
            self.model.append(nn.ReLU())
            self.model.append(nn.Dropout())

        # output layer
        self.model.append(nn.Linear(hidden_shape[-1], output_size))
        self.model.append(nn.ReLU())

    def forward(self, x):
        return self.model(x)


class SalesLstm(nn.Module):

    def __init__(self, lstm_architecture: dict[str, int],
                 nn_architecture: list[int], fcn_architecture: list[int],
                 dropout_prop=0.5):
        """
        :param lstm_architecture: dict of lstm parameters input_size,hidden_size,num_layers
        :param nn_architecture: list of nn_architecture [input_size,hidden1,hidden2..,output]
        :param fcn_architecture: list of fcn_architecture [hidden1,hidden2..,output] input_size=nn_hidden[-1]+lstm_hidden
        :param dropout_prop: probability of the dropout
        """
        super(SalesLstm, self).__init__()
        self.lstm = nn.LSTM(input_size=lstm_architecture['input_size'], hidden_size=lstm_architecture['hidden_size'],
                            num_layers=lstm_architecture['num_layers'], batch_first=True, dropout=dropout_prop)

        self.store_data_nn = build_seq_nn(nn_architecture, dropout_prop)

        fcn_architecture = [lstm_architecture['hidden_size'] + nn_architecture[-1]] + fcn_architecture
        self.fcn = build_seq_nn(fcn_architecture, dropout_prop)
        self.lstm_architecture = lstm_architecture
        self.fcn_architecture = fcn_architecture
        self.nn_architecture = nn_architecture

    def _init_hidden(self, batch_size):
        param_iter = self.parameters()
        lstm_weights = next(param_iter)

        hidden = (
            lstm_weights.new(self.lstm_architecture['num_layers'], batch_size,
                             self.lstm_architecture['hidden_size']).zero_(),
            lstm_weights.new(self.lstm_architecture['num_layers'], batch_size,
                             self.lstm_architecture['hidden_size']).zero_()
        )
        return hidden

    def forward(self, lstm_in, nn_in, lstm_hidden=None):
        # lstm in --> batch_size , seq_size , input_size
        # lstm out --> batch_size , seq_size , hidden_size
        batch_size = lstm_in.shape[0]
        if lstm_hidden is None:
            lstm_hidden = self._init_hidden(batch_size)

        lstm_out, lstm_hidden = self.lstm(lstm_in, lstm_hidden)

        # get the last output of the sequence
        lstm_out = lstm_out[:, -1, :]

        lstm_out = lstm_out.view(lstm_out.shape[0], -1)

        # store_data_nn in batch_size , input_size
        # store_data_nn out batch_size , hidden_size
        store_data_out = self.store_data_nn(nn_in)

        fcn_in = torch.cat((lstm_out, store_data_out), dim=1)

        fcn_out = self.fcn(fcn_in)

        return fcn_out, lstm_hidden


def build_seq_nn(architecture: list, dropout_prop):
    # input layer
    model = nn.Sequential(nn.Linear(architecture[0], architecture[1]), nn.ReLU(), nn.Dropout(dropout_prop))

    # hidden layers
    for i in range(1, len(architecture) - 2):
        model.append(nn.Linear(architecture[i], architecture[i + 1]))
        model.append(nn.ReLU())
        model.append(nn.Dropout())

    # output layer
    model.append(nn.Linear(architecture[-2], architecture[-1]))
    model.append(nn.ReLU())
    return model
