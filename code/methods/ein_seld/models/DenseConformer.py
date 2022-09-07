import torch
import torch.nn as nn
from methods.utils.model_utilities import init_layer
from methods.utils.conformer.encoder import ConformerBlocks
from methods.utils.dense_block import _DenseBlock, _Transition

class DenseConformer(nn.Module):
    def __init__(self, cfg, dataset):
        super().__init__()
        self.pe_enable = False  # Ture | False
        self.cfg = cfg
        self.num_classes = dataset.num_classes

        if cfg['data']['audio_feature'] in ['logmelIV', 'salsa', 'salsalite']:
            self.sed_in_channels = 4
            self.doa_in_channels = 7
        elif cfg['data']['audio_feature'] in ['logmelgcc']:
            self.sed_in_channels = 4
            self.doa_in_channels = 10

        growth_rate = (16, 24, 32, 40)
        num_layers = 4
        drop_rate = 0.
        self.sed_dense_block1 = nn.Sequential(
            _DenseBlock(num_layers=num_layers, num_input_features=self.sed_in_channels,
             bn_size=4, growth_rate=growth_rate[0], drop_rate=drop_rate),
            _Transition(num_input_features=growth_rate[0]*(num_layers)+self.sed_in_channels, num_output_features=growth_rate[0])
        )
        self.sed_dense_block2 = nn.Sequential(
            _DenseBlock(num_layers=num_layers, num_input_features=growth_rate[0],
             bn_size=4, growth_rate=growth_rate[1], drop_rate=drop_rate),
            _Transition(num_input_features=growth_rate[1]*(num_layers)+growth_rate[0], num_output_features=growth_rate[1])
        )
        self.sed_dense_block3 =nn.Sequential(
            _DenseBlock(num_layers=num_layers, num_input_features=growth_rate[1],
             bn_size=4, growth_rate=growth_rate[2], drop_rate=drop_rate),
            _Transition(num_input_features=growth_rate[2]*(num_layers)+growth_rate[1], num_output_features=growth_rate[2])
        )
        self.sed_dense_block4 = nn.Sequential(
            _DenseBlock(num_layers=num_layers, num_input_features=growth_rate[2],
             bn_size=4, growth_rate=growth_rate[3], drop_rate=drop_rate),
            nn.BatchNorm2d(num_features=growth_rate[3]*(num_layers)+growth_rate[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=growth_rate[3]*(num_layers)+growth_rate[2], out_channels=256,
            kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.doa_dense_block1 = nn.Sequential(
            _DenseBlock(num_layers=num_layers, num_input_features=self.doa_in_channels,
             bn_size=4, growth_rate=growth_rate[0], drop_rate=drop_rate),
            _Transition(num_input_features=growth_rate[0]*(num_layers)+self.doa_in_channels, num_output_features=growth_rate[0])
        )
        self.doa_dense_block2 = nn.Sequential(
            _DenseBlock(num_layers=num_layers, num_input_features=growth_rate[0],
             bn_size=4, growth_rate=growth_rate[1], drop_rate=drop_rate),
            _Transition(num_input_features=growth_rate[1]*(num_layers)+growth_rate[0], num_output_features=growth_rate[1])
        )
        self.doa_dense_block3 =nn.Sequential(
            _DenseBlock(num_layers=num_layers, num_input_features=growth_rate[1],
             bn_size=4, growth_rate=growth_rate[2], drop_rate=drop_rate),
            _Transition(num_input_features=growth_rate[2]*(num_layers)+growth_rate[1], num_output_features=growth_rate[2])
        )
        self.doa_dense_block4 = nn.Sequential(
            _DenseBlock(num_layers=num_layers, num_input_features=growth_rate[2],
             bn_size=4, growth_rate=growth_rate[3], drop_rate=drop_rate),
            nn.BatchNorm2d(num_features=growth_rate[3]*(num_layers)+growth_rate[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=growth_rate[3]*(num_layers)+growth_rate[2], out_channels=256,
            kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.stitch = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(growth_rate[0], 2, 2).uniform_(0.1, 0.9)),
            nn.Parameter(torch.FloatTensor(growth_rate[1], 2, 2).uniform_(0.1, 0.9)),
            nn.Parameter(torch.FloatTensor(growth_rate[2], 2, 2).uniform_(0.1, 0.9)),
            nn.Parameter(torch.FloatTensor(256, 2, 2).uniform_(0.1, 0.9)),
            nn.Parameter(torch.FloatTensor(256, 2, 2).uniform_(0.1, 0.9)),
            nn.Parameter(torch.FloatTensor(256, 2, 2).uniform_(0.1, 0.9))
        ])
        
        self.sed_conformer_track1 = ConformerBlocks(encoder_dim=256, num_layers=2)
        self.sed_conformer_track2 = ConformerBlocks(encoder_dim=256, num_layers=2)
        self.sed_conformer_track3 = ConformerBlocks(encoder_dim=256, num_layers=2)

        self.doa_conformer_track1 = ConformerBlocks(encoder_dim=256, num_layers=2)
        self.doa_conformer_track2 = ConformerBlocks(encoder_dim=256, num_layers=2)
        self.doa_conformer_track3 = ConformerBlocks(encoder_dim=256, num_layers=2)
        
        self.fc_sed_track1 = nn.Linear(256, self.num_classes, bias=True)
        self.fc_sed_track2 = nn.Linear(256, self.num_classes, bias=True)
        self.fc_sed_track3 = nn.Linear(256, self.num_classes, bias=True)
        self.fc_doa_track1 = nn.Linear(256, 3, bias=True)
        self.fc_doa_track2 = nn.Linear(256, 3, bias=True)
        self.fc_doa_track3 = nn.Linear(256, 3, bias=True)
        self.final_act_sed = nn.Sequential() # nn.Sigmoid()
        self.final_act_doa = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.init_weight()
    
    def init_weight(self):
        init_layer(self.fc_sed_track1)
        init_layer(self.fc_sed_track2)
        init_layer(self.fc_sed_track3)
        init_layer(self.fc_doa_track1)
        init_layer(self.fc_doa_track2)
        init_layer(self.fc_doa_track3)

    def forward(self, x):
        """
        x: spectrogram, (batch_size, num_channels, num_frames, num_freqBins)
        """
        x_sed = x[:, :self.sed_in_channels]
        x_doa = x

        # cnn
        x_sed = self.sed_dense_block1(x_sed)
        x_doa = self.doa_dense_block1(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch[0][:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[0][:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch[0][:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[0][:, 1, 1], x_doa)
        x_sed = self.sed_dense_block2(x_sed)
        x_doa = self.doa_dense_block2(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch[1][:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[1][:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch[1][:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[1][:, 1, 1], x_doa)
        x_sed = self.sed_dense_block3(x_sed)
        x_doa = self.doa_dense_block3(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch[2][:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[2][:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch[2][:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[2][:, 1, 1], x_doa)
        x_sed = self.sed_dense_block4(x_sed)
        x_doa = self.doa_dense_block4(x_doa)
        x_sed = x_sed.mean(dim=3) # (N, C, T)
        x_doa = x_doa.mean(dim=3) # (N, C, T)

        # Conformer
        x_sed = x_sed.permute(0, 2, 1) # (N, T, C)
        x_doa = x_doa.permute(0, 2, 1) # (N, T, C)
        x_sed_1 = self.sed_conformer_track1(x_sed) # (N, T, C)
        x_doa_1 = self.doa_conformer_track1(x_doa) # (N, T, C)
        x_sed_1 = torch.einsum('c, ntc -> ntc', self.stitch[3][:, 0, 0], x_sed_1) + \
            torch.einsum('c, ntc -> ntc', self.stitch[3][:, 0, 1], x_doa_1)
        x_doa_1 = torch.einsum('c, ntc -> ntc', self.stitch[3][:, 1, 0], x_sed_1) + \
            torch.einsum('c, ntc -> ntc', self.stitch[3][:, 1, 1], x_doa_1)
        
        x_sed_2 = self.sed_conformer_track2(x_sed) # (N, T, C)   
        x_doa_2 = self.doa_conformer_track2(x_doa) # (N, T, C)
        x_sed_2 = torch.einsum('c, ntc -> ntc', self.stitch[4][:, 0, 0], x_sed_2) + \
            torch.einsum('c, ntc -> ntc', self.stitch[4][:, 0, 1], x_doa_2)
        x_doa_2 = torch.einsum('c, ntc -> ntc', self.stitch[4][:, 1, 0], x_sed_2) + \
            torch.einsum('c, ntc -> ntc', self.stitch[4][:, 1, 1], x_doa_2)

        x_sed_3 = self.sed_conformer_track3(x_sed) # (N, T, C)
        x_doa_3 = self.doa_conformer_track3(x_doa) # (N, T, C)
        x_sed_3 = torch.einsum('c, ntc -> ntc', self.stitch[5][:, 0, 0], x_sed_3) + \
            torch.einsum('c, ntc -> ntc', self.stitch[5][:, 0, 1], x_doa_3)
        x_doa_3 = torch.einsum('c, ntc -> ntc', self.stitch[5][:, 1, 0], x_sed_3) + \
            torch.einsum('c, ntc -> ntc', self.stitch[5][:, 1, 1], x_doa_3)

        # fc
        x_sed_1 = self.final_act_sed(self.fc_sed_track1(x_sed_1))
        x_sed_2 = self.final_act_sed(self.fc_sed_track2(x_sed_2))
        x_sed_3 = self.final_act_sed(self.fc_sed_track3(x_sed_3))
        x_sed = torch.stack((x_sed_1, x_sed_2, x_sed_3), 2)
        x_doa_1 = self.final_act_doa(self.fc_doa_track1(x_doa_1))
        x_doa_2 = self.final_act_doa(self.fc_doa_track2(x_doa_2))
        x_doa_3 = self.final_act_doa(self.fc_doa_track3(x_doa_3))
        x_doa = torch.stack((x_doa_1, x_doa_2, x_doa_3), 2)
        output = {
            'sed': x_sed,
            'doa': x_doa,
        }

        return output

