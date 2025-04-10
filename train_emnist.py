import numpy as np
import os
import torch
import pytorch_lightning as pl
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.entry_detection.resnet import ResNet
from src.entry_detection.generate_box_dataset import get_box_dataset


class BoxDataset(Dataset):
    def __init__(self, data, labels):
        
        super().__init__()
        
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
def filter_labels(labels, data, indices):
    l = []
    d = []
    for label in indices:
        d.append(data[labels == label])
        l.append(labels[labels == label])
    data = np.concatenate(d)
    labels = np.concatenate(l)
    return labels, data
    
def split_indices(labels, fraction):
    train_indices = []
    val_indices = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        lbls = np.flatnonzero(labels == label)
        np.random.shuffle(lbls)
        train_indices.append(lbls[: int(fraction * lbls.shape[0])])
        val_indices.append(lbls[int(fraction * lbls.shape[0]):])
    
    train_indices = np.concatenate(train_indices)
    val_indices = np.concatenate(val_indices)
    return train_indices, val_indices

def one_hot(labels):
    unique = np.unique(labels)
    result = np.zeros((labels.shape[0], unique.shape[0]), dtype=labels.dtype)
    for i, label in enumerate(np.unique(labels)):
        for j in (labels == label):
            result[i, j] = 1
    return result

def split_data(X, y, yh, train_fraction, val_fraction, test_fraction, nb_classes):
    
    tot_idxs = (y == 1).sum()
    idxs = np.flatnonzero(y == 0)
    np.random.shuffle(idxs)
    idxs_zero = idxs[tot_idxs:]
    idxs = [i not in idxs_zero for i in range(y.shape[0])]
    X = X[idxs]
    y = y[idxs]
    
    random_permutation = np.random.permutation(np.arange(len(X)))
    X = np.array(X)
    X = np.swapaxes(X, axis1=1, axis2=2)
    X = np.swapaxes(X, axis1=1, axis2=3)
    y = [int(i) for i in y]
    
    train_data = X[random_permutation[: int(np.round(X.shape[0] * train_fraction))]]
    val_data = X[random_permutation[int(np.round(X.shape[0] * train_fraction)): int(np.round(X.shape[0] * (train_fraction + val_fraction)))]]
    test_data = X[random_permutation[int(np.round(X.shape[0] * (train_fraction + val_fraction))):]]
    
    y_temp = np.zeros((len(y), nb_classes))
    for i, y_i in enumerate(y):
        y_temp[i, y_i] = 1
    
    counts = np.zeros((nb_classes,))
    for i in random_permutation[: int(np.round(X.shape[0] * train_fraction))]:
        counts[y[i]] += 1
        
    train_y = y_temp[random_permutation[: int(np.round(X.shape[0] * train_fraction))]]
    val_y = y_temp[random_permutation[int(np.round(X.shape[0] * train_fraction)): int(np.round(X.shape[0] * (train_fraction + val_fraction)))]]
    test_y = y_temp[random_permutation[int(np.round(X.shape[0] * (train_fraction + val_fraction))):]]
    
    return train_data, val_data, test_data, train_y, val_y, test_y, counts

if __name__ == "__main__":
        
    characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
    data = np.genfromtxt('./data/emnist_balanced/emnist-balanced-test.csv', delimiter=',')
    labels = data[:, 0]
    data = data[:, 1:].reshape((data.shape[0], 28, 28))
        
    print(data.shape)
    data = np.swapaxes(data, 1, 2)
    
    np.save("./data/emnist_balanced/emnist-balanced-test.npy", data)
    np.save("./data/emnist_balanced/emnist-balanced-test_labels.npy", np.int32(labels))
    
    data = np.genfromtxt('./data/emnist_balanced/emnist-balanced-train.csv', delimiter=',')
    labels = data[:, 0]
    data = data[:, 1:].reshape((data.shape[0], 28, 28))
    
    data = np.swapaxes(data, 1, 2)
    
    labels = np.int32(labels)
    
    data = data.astype(np.float32)
    labels = labels.astype(np.float32)
    
    data = data.reshape((data.shape[0], 1, data.shape[1], data.shape[2]))
    
    indices = np.arange(1, 17)
    labels, data = filter_labels(labels, data, indices)
    
    fraction = 0.8
    train_indices, val_indices = split_indices(labels, fraction=fraction)
    
    train = data[train_indices]
    val = data[val_indices]
    
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    
    test = np.load("./data/emnist_balanced/emnist-balanced-test.npy")
    test_labels = np.load("./data/emnist_balanced/emnist-balanced-test_labels.npy")
    
    test = test.reshape((test.shape[0], 1, test.shape[1], test.shape[2]))
    
    test_labels, test = filter_labels(test_labels, test, indices)
    
    test = test.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    
    train_labels = one_hot(train_labels)
    val_labels = one_hot(val_labels)
    test_labels = one_hot(test_labels)
    
    train_fraction = 0.7
    val_fraction = 0.2
    test_fraction = 0.1
    path = f'{os.getcwd()}/data/sudokus'
    X, y, yh = get_box_dataset(path)
    
    train_data, val_data, test_data, train_y, val_y, test_y, counts = split_data(X, y, yh, train_fraction, val_fraction, test_fraction, train_labels.shape[1])
    
    train = np.vstack([train, train_data])
    train_labels = np.vstack([train_labels, train_y])
    val = np.vstack([val, val_data])
    val_labels = np.vstack([val_labels, val_y])
    test = np.vstack([test, test_data])
    test_labels = np.vstack([test_labels, test_y])
    
    train_dataset = BoxDataset(train, train_labels)
    val_dataset = BoxDataset(val, val_labels)
    test_dataset = BoxDataset(test, test_labels)
    
    device = [torch.device(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [torch.device("cpu")]
    accelerator = "gpu" if device[0] != torch.device("cpu") else "cpu"
    
    epochs = 100
    batch_size = 256

    train_loader = DataLoader(train_dataset, num_workers=4, batch_size=batch_size, shuffle=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, num_workers=4, batch_size=batch_size, persistent_workers=True)
    test_loader = DataLoader(test_dataset, num_workers=4, batch_size=batch_size, persistent_workers=True)

    channels = [64, 64, 128, 256, 512]
    layers = [3, 4, 6, 4]
    expansion = 4
    first_layer = [nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1)]
    padding = 1
    stride = 2
    lr = 0.001
    bottleneck_patch_size = 4
    h_bottleneck = 512
    h = train_labels.shape[1]
    print(h)
    loss = nn.CrossEntropyLoss()
    model = ResNet(channels, layers, expansion, first_layer, padding, stride, bottleneck_patch_size, h_bottleneck, h, batch_size, lr, loss=loss)
    
    logger = TensorBoardLogger("tb_logs/sudoku", name=f'resnet_box_{h}')
    callbacks = [ModelCheckpoint(monitor='val_loss', save_top_k=1, save_last=True)]
    trainer = pl.Trainer(max_epochs=epochs, logger=logger, accelerator=accelerator, strategy='auto', devices='auto', callbacks=callbacks)
                            
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=model, dataloaders=test_loader)

