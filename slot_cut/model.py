from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from cut.st import STCut

from slot_cut.utils import Tensor
from slot_cut.utils import assert_shape
from slot_cut.utils import build_grid
from slot_cut.utils import conv_transpose_out_shape
from slot_cut.resnet import ResNetEncoder


class SlotCut(nn.Module):
    def __init__(self, in_features, num_slots, slot_size, mlp_hidden_size, epsilon=1e-8, gamma=0.5, cuda=True):
        super().__init__()
        self.in_features = in_features
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.fg_temp = 0.5
        self.bg_temp = 0.1

        input_res = 64
        self.in_pos_embedding = SoftPositionEmbed(3, in_features, (input_res,input_res))

        self.norm_inputs = nn.LayerNorm(in_features)

        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        self.st_cut = STCut(N=64, gamma=gamma)
        self.st_cut.build_solver()

        self.n_edge_channels = 6
        self.K = 1
        self.N = input_res
        self.gamma = gamma
        self.DEVICE = torch.device("cuda" if cuda else "cpu")

        self.weights_net = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=100, out_channels=self.n_edge_channels * (self.num_slots-1), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.weights_net_fg = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=100, out_channels=self.n_edge_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )


        self.k_net2 = nn.Sequential(
            nn.Linear(in_features, self.mlp_hidden_size*2),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size*2, self.slot_size),
        )

        self.k_net2_2 = nn.Sequential(
            nn.Linear(in_features, self.mlp_hidden_size*2),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size*2, self.slot_size),
        )

        self.k_net2_3 = nn.Sequential(
            nn.Linear(in_features, self.mlp_hidden_size*2),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size*2, self.slot_size),
        )


    def forward(self, inputs: Tensor, img):
        #`inputs` has shape: [batch_size, filter_size, height, width]

        inputs = self.in_pos_embedding(inputs)
        inputs_permed = inputs.permute(0,2,3,1)
        inputs_permed = self.norm_inputs(inputs_permed)

        c_params = self.weights_net(inputs)
        c_params_fg = self.weights_net_fg(inputs)

        csh = c_params.shape
        c_params = c_params.reshape((csh[0]*(self.num_slots-1),1, self.n_edge_channels, self.N, self.N))
        c_params_fg = c_params_fg.reshape((csh[0],1, self.n_edge_channels, self.N, self.N))


        x = self.st_cut.solve(c_params)
        mask = self.st_cut.get_node_vars(x)

        x_fg = self.st_cut.solve(c_params_fg)

        mask_fg = self.st_cut.get_edge_vars_st(x_fg)
        mask_fg = mask_fg.reshape((csh[0], 1, self.N, self.N))
        mask_bg = 1-mask_fg

        mask = mask.reshape((csh[0], self.num_slots-1, self.N, self.N))
        mask = F.softmax(mask/self.fg_temp, dim=1)

        mask = mask_fg*mask
        mask = torch.cat([mask_bg, mask], dim=1)

        mask = mask.reshape((csh[0], self.num_slots, self.N, self.N))
        #temp hack
        mask = F.softmax(mask/self.bg_temp, dim=1)

        k = inputs_permed
        k1 = self.k_net2(k)
        k2 = self.k_net2_2(k)
        k3 = self.k_net2_3(k)

        slots1 = (mask.unsqueeze(4)*k1.unsqueeze(1)).mean(dim=[2,3])
        slots2 = (mask.unsqueeze(4)*k2.unsqueeze(1)).mean(dim=[2,3])
        slots3 = (mask.unsqueeze(4)*k3.unsqueeze(1)).mean(dim=[2,3])

        slots = slots1
        slots= slots + self.mlp((slots))
        slots = slots + self.mlp2(slots2)
        slots = slots + self.mlp3(slots3)

        return slots


class SlotCutModel(nn.Module):
    def __init__(
        self,
        resolution: Tuple[int, int],
        num_slots: int,
        in_channels: int = 3,
        kernel_size: int = 5,
        slot_size: int = 64,
        hidden_dims: Tuple[int, ...] = (64, 64, 64, 64),
        decoder_resolution: Tuple[int, int] = (8, 8),
        empty_cache=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.slot_size = slot_size
        self.empty_cache = empty_cache
        self.hidden_dims = hidden_dims
        self.decoder_resolution = decoder_resolution
        self.out_features = self.hidden_dims[-1]

        #Resnet encoder
        self.encoder = ResNetEncoder()

        # Build Decoder
        modules = []

        in_size = decoder_resolution[0]
        out_size = in_size

        for i in range(len(self.hidden_dims) - 1, -1, -1): #128
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hidden_dims[i],
                        self.hidden_dims[i - 1],
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            out_size = conv_transpose_out_shape(out_size, 2, 2, 5, 1)

        assert_shape(
            resolution,
            (out_size, out_size),
            message="Output shape of decoder did not match input resolution. Try changing `decoder_resolution`.",
        )

        modules.append(
              nn.Sequential(
                  nn.ConvTranspose2d(
                      self.out_features, self.out_features, kernel_size=5, stride=1, padding=2, output_padding=0,
                  ),
                  nn.LeakyReLU(),

                  nn.ConvTranspose2d(
                      self.out_features, self.out_features, kernel_size=5, stride=1, padding=2, output_padding=0,
                  ),
                  nn.LeakyReLU(),

                  nn.ConvTranspose2d(self.out_features, 4, kernel_size=3, stride=1, padding=1, output_padding=0,),
              )
          )

        assert_shape(resolution, (out_size, out_size), message="")

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, self.decoder_resolution)

        self.slot_cut = SlotCut(
            in_features=100,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=128,
        )

    def forward(self, x):
        if self.empty_cache:
            torch.cuda.empty_cache()

        batch_size, num_channels, height, width = x.shape
        encoder_out = self.encoder(x)

        slots = self.slot_cut(encoder_out, x)
        assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))
        # `slots` has shape: [batch_size, num_slots, slot_size].
        batch_size, num_slots, slot_size = slots.shape

        slots = slots.view(batch_size * num_slots, slot_size, 1, 1)
        decoder_in = slots.repeat(1, 1, self.decoder_resolution[0], self.decoder_resolution[1])

        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)

        assert_shape(out.size(), (batch_size * num_slots, num_channels + 1, height, width))
        out = out.view(batch_size, num_slots, num_channels + 1, height, width)

        recons = out[:, :, :num_channels, :, :]

        masks = out[:, :, -1:, :, :]
        masks = F.softmax(masks, dim=1)
        recons = torch.tanh(recons)

        recon_combined = torch.sum(recons * masks, dim=1)
        return recon_combined, recons, masks, slots

    def loss_function(self, input):
        recon_combined, recons, masks, slots = self.forward(input)
        loss = F.mse_loss(recon_combined, input)
        return {
            "loss": loss,
        }


class SoftPositionEmbed(nn.Module):
    def __init__(self, num_channels: int, hidden_size: int, resolution: Tuple[int, int]):
        super().__init__()
        self.dense = nn.Linear(in_features=num_channels + 1, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs: Tensor):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        assert_shape(inputs.shape[1:], emb_proj.shape[1:])
        return inputs + emb_proj
