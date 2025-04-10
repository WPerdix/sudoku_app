import torch
import pytorch_lightning as pl

from torch import nn, optim


class AutoEncoder(pl.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, batch_size: int=128, lr: float=1e-3, lr_min: float=1e-5, lr_factor: float=0.2, lr_patience: int=10, loss=nn.functional.mse_loss) -> None:
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self._trainer = 1
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        
        # lr scheduler paramters
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, x):
        return self.decoder(self.encoder(x))
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. it is independent of forward
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        train_loss = self.loss(x_hat, y)
        self.training_step_outputs.append({'train_loss': train_loss.item()})
        return train_loss
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = self.loss(x_hat, y)
        self.test_step_outputs.append({'test_loss': test_loss.item()})
        
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = self.loss(x_hat, y)
        self.validation_step_outputs.append({'val_loss': val_loss.item()})

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         factor=self.lr_factor,
                                                         patience=self.lr_patience,
                                                         min_lr=self.lr_min)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, 'monitor': 'val_loss'}
    
    def predict_step(self, batch, batch_idx):
        return self(batch)
    
    def on_train_epoch_end(self):            
        # calculating average loss  
        avg_loss = torch.Tensor([x['train_loss'] for x in self.training_step_outputs]).to(self.device).mean().item()
        
        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                        avg_loss,
                                        self.current_epoch)
        print("\nTraining loss: ", avg_loss)

        self.log('train_loss', avg_loss)
            
        self.training_step_outputs.clear()
        
    def on_validation_epoch_end(self):            
        # calculating average loss  
        avg_loss = torch.Tensor([x['val_loss'] for x in self.validation_step_outputs]).to(self.device).mean().item()
        
        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Val",
                                          avg_loss,
                                          self.current_epoch)
        print("\nValidation loss: ", avg_loss)

        self.log('val_loss', avg_loss)
            
        self.validation_step_outputs.clear()
        
    def on_test_epoch_end(self) -> None:
        # calculating average loss  
        avg_loss = torch.Tensor([x['test_loss'] for x in self.test_step_outputs]).to(self.device).mean().item()
        
        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Test",
                                          avg_loss,
                                          self.current_epoch)
        print("\nTest loss: ", avg_loss)

        self.log('test_loss', avg_loss)
            
        self.test_step_outputs.clear()
        
    @property
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        trainable_num_p = int(sum([p.numel() for p in model_parameters]))
        total_num_p = int(sum([p.numel() for p in self.parameters()]))
        non_trainable_num_p = total_num_p - trainable_num_p
        print(f'{"Total params:":<25}{total_num_p:>15,}')
        print(f'{"Trainable params:":<25}{trainable_num_p:>15,}')
        print(f'{"Non-trainable params:":<25}{non_trainable_num_p:>15,}')
        return trainable_num_p, total_num_p, non_trainable_num_p
    
