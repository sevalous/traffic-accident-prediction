# model creation and training
from torch import Tensor
from torch.nn import LSTM, Linear
from torch.utils.data import Dataset, DataLoader
from coral_pytorch.dataset import corn_label_from_logits
# evaluation metrics
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError

from pytorch_lightning import LightningDataModule, LightningModule

class RoadAccidentDataset(Dataset):
    def __init__(self, X:Tensor, y:Tensor):
        self.X = X.float() # convert to float to ensure the dtypes are consistent
        self.y = y.float() # still necessary even if they are already floats, e.g., float32 is different to float64

    def __len__(self): # for getting the number of instances
        return len(self.X)

    def __getitem__(self, idx): # for getting a specific set of values at a certain index
        return self.X[idx].unsqueeze(0), self.y[idx]
        # the above converts the features into shape (batch_size, seq_len, number of features)
        # the true labels, y, remains the same shape
  
class RoadAccidentDataModule(LightningDataModule):
    def __init__(self, X_train, X_test, X_val, y_train, y_val, y_test, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

    def train_dataloader(self):
        train_dataset = RoadAccidentDataset(self.X_train, self.y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader

    def val_dataloader(self):
        validation_dataset = RoadAccidentDataset(self.X_val, self.y_val)
        validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)

        return validation_dataloader

    def test_dataloader(self):
        test_dataset = RoadAccidentDataset(self.X_test, self.y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return test_dataloader
        
class AccidentLikelihoodNetwork(LightningModule):
    def __init__(self, n_features, hidden_size, output_size, batch_size,
                    num_layers, learning_rate,
                    optimiser, criterion, dropout):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.optimiser = optimiser
        self.dropout = dropout
        self.criterion = criterion
        
        self.train_mae = MeanAbsoluteError()
        self.train_mae_macroaveraged_epoch = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.val_mae_macroaveraged_epoch = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()

        self.train_mape = MeanAbsolutePercentageError()
        self.val_mape = MeanAbsolutePercentageError()
        self.test_mape = MeanAbsolutePercentageError()
        
        self.lstm = LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout)
        self.classifier = Linear(hidden_size, output_size-1) # -1 due to using ordinal regression
        # Linear is fully connected layer
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.classifier(lstm_out[:, -1, :])
        # lstm_out[:, -1, :] takes the last time step and uses it for the classifier
        return output

    def _shared_step(self, batch):
        # features is x, true_labels is y
        features, true_labels = batch
        logits=self.forward(features)

        # calculate Log Loss
        loss = self.criterion(logits, true_labels, num_classes=self.output_size)

        predicted_labels = corn_label_from_logits(logits)

        return loss, true_labels, predicted_labels

    def configure_optimizers(self):
        optimiser = self.optimiser(self.parameters(), lr=self.learning_rate)
        return optimiser

    def training_step(self, train_batch, train_idx):
        loss, true_labels, predicted_labels = self._shared_step(train_batch)
        train_mae = self.train_mae(predicted_labels, true_labels)
        self.train_mae_macroaveraged_epoch(predicted_labels, true_labels)
        train_mape = self.train_mape(predicted_labels, true_labels)
        train_mse = self.train_mse(predicted_labels, true_labels)
        train_rmse = train_mse**0.5 # equivalent to square root

        self.log('train_loss', loss, on_epoch=True)
        self.log('train_mse', train_mse, on_epoch=True)
        self.log('train_rmse', train_rmse, on_epoch=True)
        self.log('train_mae', train_mae, on_epoch=True)
        self.log('train_mape', train_mape, on_epoch=True)
        return loss
    
    def on_train_epoch_end(self): # for calculating the macroaveraged MAE
        self.log('train_mae_epoch', self.train_mae_macroaveraged_epoch.compute())
        self.train_mae_macroaveraged_epoch.reset()

    def validation_step(self, val_batch, val_idx):
        loss, true_labels, predicted_labels = self._shared_step(val_batch)
        val_mae = self.val_mae(predicted_labels, true_labels)
        self.val_mae_macroaveraged_epoch(predicted_labels, true_labels)
        val_mape = self.val_mape(predicted_labels, true_labels)
        val_mse = self.val_mse(predicted_labels, true_labels)
        val_rmse = val_mse**0.5

        self.log('val_loss', loss, on_epoch=True)
        self.log('val_mse', val_mse, on_epoch=True)
        self.log('val_rmse', val_rmse, on_epoch=True)
        self.log('val_mae', val_mae, on_epoch=True)
        self.log('val_mape', val_mape, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self): # for calculating the macroaveraged MAE
        self.log('val_mae_epoch', self.val_mae_macroaveraged_epoch.compute())
        self.val_mae_macroaveraged_epoch.reset()

    def test_step(self, test_batch, test_idx):
        loss, true_labels, predicted_labels = self._shared_step(test_batch)
        test_mae = self.test_mae(predicted_labels, true_labels)
        test_mape = self.test_mape(predicted_labels, true_labels)
        test_mse = self.test_mse(predicted_labels, true_labels)
        test_rmse = test_mse**0.5

        self.log('test_loss', loss, on_epoch=True)
        self.log('test_mse', test_mse, on_epoch=True)
        self.log('test_rmse', test_rmse, on_epoch=True)
        self.log('test_mae', test_mae, on_epoch=True)
        self.log('test_mape', test_mape, on_epoch=True)
        return loss