import numpy as np
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

        self.model = nn.Sequential(nn.Linear(input_size, hidden_shape[0]), nn.ReLU())

        # hidden layers
        for i in range(len(hidden_shape) - 1):
            self.model.append(nn.Linear(hidden_shape[i], hidden_shape[i + 1]))
            self.model.append(nn.ReLU())
            self.model.append(nn.Dropout(dropout_prop))

        # output layer
        self.model.append(nn.Linear(hidden_shape[-1], output_size))
        self.model.append(nn.ReLU())

    def forward(self, x):
        return self.model(x)


class SalesLstm(nn.Module):
    state_holiday_cols = ['StateHoliday_christmas', 'StateHoliday_easter', 'StateHoliday_public']
    weekdays_cols = [f"DayOfWeek_{i}" for i in range(1, 8)]
    store_type_cols = [f"StoreType_{i}" for i in ['a', 'b', 'c', 'd']]
    assortment_cols = [f"Assortment_{i}" for i in ['a', 'b', 'c']]

    lstm_sales_cols = ['Sales', 'Promo', 'SchoolHoliday', 'month', 'day'] + state_holiday_cols + weekdays_cols

    nn_sales_cols = ['Promo', 'SchoolHoliday', 'month', 'day', 'CompetitionDistance', 'Promo2', 'Promo2Since',
                     'CompetitionOpenSince',
                     'isPromoMonth'] + weekdays_cols + state_holiday_cols + store_type_cols + assortment_cols

    def __init__(self, lstm_architecture: dict,
                 fcn_architecture: dict,
                 dropout_prop=None):
        """
        :param lstm_architecture: dict of lstm parameters input_size,hidden_size,num_layers
        :param fcn_architecture: list of fcn_architecture [hidden1,hidden2..,output] input_size=nn_hidden[-1]+lstm_hidden
        :param dropout_prop: probability of the dropout
        """
        super(SalesLstm, self).__init__()
        self.lstm = nn.LSTM(input_size=lstm_architecture['input_size'], hidden_size=lstm_architecture['hidden_size'],
                            num_layers=lstm_architecture['num_layers'], batch_first=True)

        self.lstm_architecture = lstm_architecture
        self.fcn_architecture = fcn_architecture

        self.fcn = build_seq_nn(fcn_architecture['input_size'] + lstm_architecture['hidden_size'],
                                fcn_architecture['hidden_shape'], dropout_prop=dropout_prop)

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

        fcn_in = torch.cat((nn_in, lstm_out), dim=1)

        fcn_out = self.fcn(fcn_in)

        return fcn_out, lstm_hidden

    @staticmethod
    def to_tensor(df_input):
        numpy_input = df_input.to_numpy()
        numpy_input = np.expand_dims(numpy_input, axis=0)
        return torch.tensor(numpy_input, dtype=torch.float)

    def predict(self, prev_data, new_data, hidden=None, device='cpu'):
        lstm_data = self.to_tensor(prev_data[self.lstm_sales_cols])
        lstm_data=lstm_data.to(device)

        nn_data = self.to_tensor(new_data[self.nn_sales_cols])
        nn_data = nn_data.to(device)
        self.eval()

        with torch.no_grad():
            out, hidden = self.forward(lstm_data, nn_data, hidden)
        return out, hidden


def build_seq_nn(input_size, hidden_shape: list, dropout_prop=None, output_size=1):
    # hidden layer
    model = nn.Sequential(nn.Linear(input_size, hidden_shape[0]), nn.ReLU())

    # hidden layers
    for i in range(len(hidden_shape) - 1):
        model.append(nn.Linear(hidden_shape[i], hidden_shape[i + 1]))
        model.append(nn.ReLU())
        if dropout_prop is not None:
            model.append(nn.Dropout(dropout_prop))

    # output layer
    model.append(nn.Linear(hidden_shape[-1], output_size))
    model.append(nn.ReLU())
    return model
