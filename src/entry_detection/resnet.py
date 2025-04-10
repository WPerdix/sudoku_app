from torch import nn
from typing import List, final

from torch.nn.functional import mse_loss
from .autoencoder import AutoEncoder
from abc import abstractmethod, ABCMeta


class ResNetBlock(nn.Module, metaclass=ABCMeta):
    resize: bool
    projection: nn.Module
    bn1: nn.Module
    bn2: nn.Module
    bn3: nn.Module
    relu: nn.Module
    conv1: nn.Module
    conv2: nn.Module
    conv3: nn.Module
    
    @abstractmethod
    def __init__(self):
        super().__init__()
    
    @final
    def forward(self, x):
        
        identity = self.projection(x) if self.resize else x
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        
        return out + identity
        

class ResNetEncoderBlock(ResNetBlock):
    """ ResNet bottleneck encoder block with pre-activation, i.e. ResNetv2 """
    def __init__(self, channels_in: int, channels_out: int, padding: int=1, stride: int=2, expansion: int=4) -> None:
        """ Create ResNet encoder bottleneck block with pre-activation.

        Args:
            channels_in (int): Number of channels feeded to this block.
            channels_out (int): Number of channels going out this block.
            padding (int): Padding used in 3x3 convolution. Defaults to 1.
            stride (int): Stride used in 3x3 convolution. Defaults to 2.
            expansion (int): Expansion factor used. Defaults to 4.
        """
        super().__init__()
        
        channels_middle = round(channels_out / expansion)
        
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.bn2 = nn.BatchNorm2d(channels_middle)
        self.bn3 = nn.BatchNorm2d(channels_middle)
    
        self.relu = nn.ReLU()
        
        self.resize = channels_in != channels_out
        self.projection = nn.Conv2d(channels_in, channels_out, 1, stride) if self.resize else nn.Identity()
        
        self.conv1 = nn.Conv2d(channels_in, channels_middle, 1)
        self.conv2 = nn.Conv2d(channels_middle, channels_middle, 3, stride, padding)
        self.conv3 = nn.Conv2d(channels_middle, channels_out, 1)
    

class ResNetEncoder(nn.Module):
    def __init__(self, channels: List[int], layers: List[int], expansion: int=4, first_layers: None | List[nn.Module]=None, padding: int=1, stride: int=2) -> None:
        super().__init__()
        
        assert len(channels) == len(layers) + 1
        
        channels_in = channels[:-1]
        channels_out = channels[1:]
        expansion = expansion
        
        first_layers = first_layers if first_layers else [nn.Identity()]
        encoder: List[nn.Module] = [layer for layer in first_layers]
        
        for i, layer in enumerate(layers):
            channels = channels_in[i]
            stride = 1 if i == 0 else 2
            padding = 1 if i == 0 else 1
            for _ in range(layer):
                encoder.append(ResNetEncoderBlock(channels, channels_out[i], padding, stride, expansion))
                channels = channels_out[i]
                stride = 1
                padding = 1
        
        self.encoder = nn.Sequential(*encoder)
    
    def forward(self, x):
        return self.encoder(x)
    

class ResNetHead(nn.Module):
    def __init__(self, input_size: int, h_bottleneck: int, h: int) -> None:
        super().__init__()
        
        self.avg = nn.AvgPool2d(input_size)
        
        self.decoder = nn.Sequential(nn.Linear(h_bottleneck, h),
                                     nn.BatchNorm1d(h),
                                     nn.Softmax(dim=1))

    def forward(self, x):
        return self.decoder(self.avg(x).reshape((x.size(0), x.size(1))))

        
class ResNet(AutoEncoder):
    def __init__(self, channels: List[int], layers: List[int], expansion: int=4, first_layer: None | nn.Module=None, padding: int=1, stride: int=2, bottleneck_patch_size: int=2, h_bottleneck: int=64, h: int=512, batch_size=16, lr=0.001, loss=nn.functional.mse_loss) -> None:
        super().__init__(ResNetEncoder(channels, layers, expansion, first_layer, padding, stride), ResNetHead(bottleneck_patch_size, h_bottleneck, h), batch_size, lr, loss=loss)
        

class ResNetDoubleHead(AutoEncoder):
    def __init__(self, channels: List[int], layers: List[int], expansion: int=4, first_layer: None | nn.Module=None, padding: int=1, stride: int=2, bottleneck_patch_size: int=2, h_bottleneck: int=64, h1: int=512, h2: int=512, batch_size=16, lr=0.001, loss1=nn.functional.mse_loss, loss2=nn.functional.mse_loss) -> None:
        
        super().__init__(ResNetEncoder(channels, layers, expansion, first_layer, padding, stride), None, batch_size, lr)
        
        self.loss1 = loss1
        self.loss2 = loss2
        
        self.avg = nn.AvgPool2d(bottleneck_patch_size)
        
        self.head1 = nn.Sequential(nn.Linear(h_bottleneck, h1),
                                   nn.BatchNorm1d(h1),
                                   nn.Softmax(dim=1))
        
        self.head2 = nn.Sequential(nn.Linear(h_bottleneck, h2),
                                   nn.BatchNorm1d(h2),
                                   nn.Softmax(dim=1))
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.avg(x).reshape((x.size(0), x.size(1)))
        return self.head1(x), self.head2(x)
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. it is independent of forward
        x, y1, y2 = batch
        x = self.encoder(x)
        x = self.avg(x).reshape((x.size(0), x.size(1)))
        train_loss = self.loss1(y1, self.head1(x)) + self.loss2(y2, self.head2(x))
        self.training_step_outputs.append({'train_loss': train_loss.item()})
        return train_loss
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y1, y2 = batch
        x = self.encoder(x)
        x = self.avg(x).reshape((x.size(0), x.size(1)))
        test_loss = self.loss1(y1, self.head1(x)) + self.loss2(y2, self.head2(x))
        self.test_step_outputs.append({'test_loss': test_loss.item()})
        
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y1, y2 = batch
        x = self.encoder(x)
        x = self.avg(x).reshape((x.size(0), x.size(1)))
        val_loss = self.loss1(y1, self.head1(x)) + self.loss2(y2, self.head2(x))
        self.validation_step_outputs.append({'val_loss': val_loss.item()})

if __name__ == '__main__':   
    channels = [256, 256, 512, 1024, 2048]
    layers = [3, 4, 6, 4]
    expansion = 4
    first_layer = [nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)]
    last_layer = [nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1), nn.BatchNorm2d(1024), nn.Sigmoid()]
    padding = 1
    stride = 2
    batch_size = 16
    lr = 0.001
    encoder = ResNetEncoder(channels, layers, expansion, first_layer, padding, stride)

    import torch
    rand = torch.randn(1, 1024, 16, 16)
    result_encoder = encoder.forward(rand)
    
    print(result_encoder.size())
    
    bottleneck_patch_size = 2
    output_size = 2221
    h_bottleneck = 2048
    h = 512

    rand = torch.randn(1, 1024, 16, 16)
    resnet = ResNet(channels, layers, expansion, first_layer, padding, stride, bottleneck_patch_size, output_size, h_bottleneck, h)
    
    print(resnet(rand).size())
    
    # summary(encoder, (1024, 16, 16))
    # print('\n')
    # summary(decoder, (2048, 2, 2))
    # print('\n')
    # summary(autoencoder, (1024, 16, 16))
