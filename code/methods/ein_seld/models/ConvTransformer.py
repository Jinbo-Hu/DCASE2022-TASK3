import torch
import torch.nn as nn
from methods.utils.model_utilities import (DoubleConv, PositionalEncoding,
                                           init_layer)


class ConvTransformer(nn.Module):
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

        self.sed_conv_block1 = nn.Sequential(
            DoubleConv(in_channels=self.sed_in_channels, out_channels=64),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.sed_conv_block2 = nn.Sequential(
            DoubleConv(in_channels=64, out_channels=128),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.sed_conv_block3 = nn.Sequential(
            DoubleConv(in_channels=128, out_channels=256),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.sed_conv_block4 = nn.Sequential(
            DoubleConv(in_channels=256, out_channels=512),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )

        self.doa_conv_block1 = nn.Sequential(
            DoubleConv(in_channels=self.doa_in_channels, out_channels=64),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.doa_conv_block2 = nn.Sequential(
            DoubleConv(in_channels=64, out_channels=128),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.doa_conv_block3 = nn.Sequential(
            DoubleConv(in_channels=128, out_channels=256),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.doa_conv_block4 = nn.Sequential(
            DoubleConv(in_channels=256, out_channels=512),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )

        self.stitch = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(64, 2, 2).uniform_(0.1, 0.9)),
            nn.Parameter(torch.FloatTensor(128, 2, 2).uniform_(0.1, 0.9)),
            nn.Parameter(torch.FloatTensor(256, 2, 2).uniform_(0.1, 0.9)),
            nn.Parameter(torch.FloatTensor(512, 2, 2).uniform_(0.1, 0.9)),
            nn.Parameter(torch.FloatTensor(512, 2, 2).uniform_(0.1, 0.9)),
            nn.Parameter(torch.FloatTensor(512, 2, 2).uniform_(0.1, 0.9))
        ])

        if self.pe_enable:
            self.pe = PositionalEncoding(pos_len=100, d_model=512, pe_type='t', dropout=0.0)
        self.sed_trans_track1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2), num_layers=2)
        self.sed_trans_track2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2), num_layers=2)
        self.sed_trans_track3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2), num_layers=2)
        self.doa_trans_track1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2), num_layers=2)
        self.doa_trans_track2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2), num_layers=2)
        self.doa_trans_track3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2), num_layers=2)

        self.fc_sed_track1 = nn.Linear(512, self.num_classes, bias=True)
        self.fc_sed_track2 = nn.Linear(512, self.num_classes, bias=True)
        self.fc_sed_track3 = nn.Linear(512, self.num_classes, bias=True)
        self.fc_doa_track1 = nn.Linear(512, 3, bias=True)
        self.fc_doa_track2 = nn.Linear(512, 3, bias=True)
        self.fc_doa_track3 = nn.Linear(512, 3, bias=True)
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
        x: waveform, (batch_size, num_channels, data_length)
        """
        x_sed = x[:, :self.sed_in_channels]
        x_doa = x

        # cnn
        x_sed = self.sed_conv_block1(x_sed)
        x_doa = self.doa_conv_block1(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch[0][:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[0][:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch[0][:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[0][:, 1, 1], x_doa)
        x_sed = self.sed_conv_block2(x_sed)
        x_doa = self.doa_conv_block2(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch[1][:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[1][:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch[1][:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[1][:, 1, 1], x_doa)
        x_sed = self.sed_conv_block3(x_sed)
        x_doa = self.doa_conv_block3(x_doa)
        x_sed = torch.einsum('c, nctf -> nctf', self.stitch[2][:, 0, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[2][:, 0, 1], x_doa)
        x_doa = torch.einsum('c, nctf -> nctf', self.stitch[2][:, 1, 0], x_sed) + \
            torch.einsum('c, nctf -> nctf', self.stitch[2][:, 1, 1], x_doa)
        x_sed = self.sed_conv_block4(x_sed)
        x_doa = self.doa_conv_block4(x_doa)
        x_sed = x_sed.mean(dim=3) # (N, C, T)
        x_doa = x_doa.mean(dim=3) # (N, C, T)

        # transformer
        if self.pe_enable:
            x_sed = self.pe(x_sed)
        if self.pe_enable:
            x_doa = self.pe(x_doa)
        x_sed = x_sed.permute(2, 0, 1) # (T, N, C)
        x_doa = x_doa.permute(2, 0, 1) # (T, N, C)

        x_sed_1 = self.sed_trans_track1(x_sed).transpose(0, 1) # (N, T, C)
        x_doa_1 = self.doa_trans_track1(x_doa).transpose(0, 1) # (N, T, C)
        x_sed_1 = torch.einsum('c, ntc -> ntc', self.stitch[3][:, 0, 0], x_sed_1) + \
            torch.einsum('c, ntc -> ntc', self.stitch[3][:, 0, 1], x_doa_1)
        x_doa_1 = torch.einsum('c, ntc -> ntc', self.stitch[3][:, 1, 0], x_sed_1) + \
            torch.einsum('c, ntc -> ntc', self.stitch[3][:, 1, 1], x_doa_1)

        x_sed_2 = self.sed_trans_track2(x_sed).transpose(0, 1) # (N, T, C)   
        x_doa_2 = self.doa_trans_track2(x_doa).transpose(0, 1) # (N, T, C)
        x_sed_2 = torch.einsum('c, ntc -> ntc', self.stitch[4][:, 0, 0], x_sed_2) + \
            torch.einsum('c, ntc -> ntc', self.stitch[4][:, 0, 1], x_doa_2)
        x_doa_2 = torch.einsum('c, ntc -> ntc', self.stitch[4][:, 1, 0], x_sed_2) + \
            torch.einsum('c, ntc -> ntc', self.stitch[4][:, 1, 1], x_doa_2)

        x_sed_3 = self.sed_trans_track3(x_sed).transpose(0, 1) # (N, T, C)
        x_doa_3 = self.doa_trans_track3(x_doa).transpose(0, 1) # (N, T, C)
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

