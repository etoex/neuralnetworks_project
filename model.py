import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, input_dim=53, hidden_dim=128, output_dim=4, dropout=0.5):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        self.lstm = nn.LSTM(256, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = torch.relu(self.conv1(x))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout(x)
        x = torch.relu(self.conv3(x))
        x = self.dropout(x)

        x = x.transpose(1, 2)

        x, _ = self.lstm(x)
        x = self.fc(x)

        return x

