import torch.nn as nn

salesNN_parameters={

}
class SalesNN(nn.Module):
    def __init__(self, input_size, hidden_shape, output_size=1, dropout_prop=0.5):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_shape=hidden_shape

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
