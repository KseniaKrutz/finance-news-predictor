import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_size=2, num_layers=2, num_heads=2, hidden_dim=128):
        super(TimeSeriesTransformer, self).__init__()
        self.input_layer = nn.Linear(feature_size, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                activation='gelu'
            ),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.transformer(x)
        x = x[:, -1, :]  
        x = self.output_layer(x)
        return x

def transformer_train(data, time_step=30):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    X, y = create_dataset_transformer(scaled_data, time_step)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    model = TimeSeriesTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    # Тренировка
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = loss_fn(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model
