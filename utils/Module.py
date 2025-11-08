import copy
import torch
from torch import nn, Tensor
from typing import Optional, Callable, Union
from torch.nn import functional as F, MultiheadAttention, Dropout, LayerNorm, ModuleList, Linear


class DualTrans_Decoder(nn.Module):
    def __init__(self, d_model, num_layers, nhead, dim_feedforward, path, device, dual_decoder, complex_mask: bool = False,
                 pure_phase: bool = False):
        super().__init__()
        self.d_model = d_model
        self.complex_mask = complex_mask
        self.Embedding = Embedding(d_model)
        self.pure_phase = pure_phase
        self.dual_decoder = dual_decoder
        self.Encoder_DeEmbedding_heart = Encoder_DeEmbedding(d_model, num_layers, nhead, dim_feedforward, path,
                                                             layer_norm=True, complex_mask=complex_mask)
        self.Encoder_DeEmbedding_lung = Encoder_DeEmbedding(d_model, num_layers, nhead, dim_feedforward, path,
                                                            layer_norm=True, complex_mask=complex_mask)

        d_model_r = int(d_model * 24 // 64)
        if dual_decoder:
            self.Decoder_heart = Decoder(num_layers=5, num_channels=d_model_r, kernel_size=(3, 3),
                                         complex_mask=complex_mask)
            self.Decoder_lung = Decoder(num_layers=5, num_channels=d_model_r, kernel_size=(3, 3),
                                        complex_mask=complex_mask)

        else:
            self.Decoder = Decoder(num_layers=5, num_channels=d_model_r, kernel_size=(3, 3),
                                   complex_mask=complex_mask)

        self.activation = nn.ReLU()
        if complex_mask:
            self.output_act = nn.Tanh()
        else:
            self.output_act = nn.Sigmoid()

    def forward(self, src: Tensor, mask_time, mask_fre) -> tuple[Tensor, Tensor]:

        # notice now src is 1*F*T where F is the frequency nums and T the time nums
        src_embedded = self.Embedding(src)
        mask_heart = self.Encoder_DeEmbedding_heart(src_embedded, mask_time, mask_fre)
        mask_lung = self.Encoder_DeEmbedding_lung(src_embedded, mask_time, mask_fre)
        mask = torch.cat((mask_heart, mask_lung), dim=0)
        # now mask is 1*F*T
        mask_heart: torch.Tensor
        mask_lung: torch.Tensor
        if self.pure_phase is True:
            tmp = src[0, :, :] + 1j * src[1, :, :]
            src_input = src / (torch.abs(tmp) + 1e-9)

        else:
            src_input = src

        if self.dual_decoder:
            heart_esti = self.Decoder_heart(torch.concatenate((src_input, mask_heart), dim=0))
            lung_esti = self.Decoder_lung(torch.concatenate((src_input, mask_lung), dim=0))

        else:
            heart_esti = self.Decoder(torch.concatenate((src_input, mask_heart), dim=0))
            lung_esti = self.Decoder(torch.concatenate((src_input, mask_lung), dim=0))


        return mask, torch.concatenate((heart_esti, lung_esti), dim=0)


class DualTrans(nn.Module):
    def __init__(self, d_model, num_layers, nhead, dim_feedforward, path, device, complex_mask: bool = False,
                 pure_phase: bool = False):
        super().__init__()
        self.d_model = d_model
        self.complex_mask = complex_mask
        self.Embedding = Embedding(d_model)
        self.device = device
        self.Encoder_DeEmbedding_heart = Encoder_DeEmbedding(d_model, num_layers, nhead, dim_feedforward, path,
                                                             layer_norm=True, complex_mask=complex_mask)
        self.Encoder_DeEmbedding_lung = Encoder_DeEmbedding(d_model, num_layers, nhead, dim_feedforward, path,
                                                            layer_norm=True, complex_mask=complex_mask)

        self.activation = nn.ReLU()
        if complex_mask:
            self.output_act = nn.Tanh()
        else:
            self.output_act = nn.Sigmoid()

    def forward(self, src: Tensor, mask_time, mask_fre) -> tuple[Tensor, Tensor]:

        # notice now src is 1*F*T where F is the frequency nums and T the time nums
        src_embedded = self.Embedding(src)
        mask_heart = self.Encoder_DeEmbedding_heart(src_embedded, mask_time, mask_fre)
        mask_lung = self.Encoder_DeEmbedding_lung(src_embedded, mask_time, mask_fre)
        mask = torch.cat((mask_heart, mask_lung), dim=0)
        # now mask is 1*F*T
        return mask, torch.zeros((4, 51, 512), dtype=torch.float).to(device=self.device)


class Encoder_DeEmbedding(nn.Module):
    def __init__(self, d_model, num_layers, nhead, dim_feedforward, path, layer_norm: bool = True,
                 complex_mask: bool = False):
        super().__init__()
        self.d_model = d_model
        self.path = path
        self.complex_mask = complex_mask
        self.DeEmbedding = DeEmbedding(d_model, complex_mask=complex_mask)
        self.encoder_path1 = Encoder(num_layers, d_model, nhead, dim_feedforward, layer_norm=layer_norm)
        self.encoder_path2 = Encoder(num_layers, d_model, nhead, dim_feedforward, layer_norm=layer_norm)
        self.activation = nn.ReLU()
        if complex_mask:
            self.output_act = nn.Tanh()
        else:
            self.output_act = nn.Sigmoid()
        # self.Embedding_Decoder = Decoder(num_layers=3, num_channels=d_model, kernel_size=(3, 3),
        #                                 complex_mask=complex_mask)

    def forward(self, src: Tensor, mask_time, mask_fre) -> tuple[Tensor, Tensor]:

        # notice now src is 1*F*T where F is the frequency nums and T the time nums
        # src_embedded = self.Embedding(src)
        src_embedded = src
        # now src is (2*d_model)*F*T
        # notice transformer should be fed S*N*E where S is the sequence length and E the embedding
        if self.path == 'Dual':
            src_path1 = self.activation(self.encoder_path1(
                torch.transpose(src_embedded, 0, 2), mask=mask_time))
            src_path1 = torch.transpose(src_path1, 0, 2)
            # now src_time is T*F*d_model
            src_path2 = torch.transpose(src_embedded, 0, 2)
            src_path2 = self.activation(self.encoder_path2(
                torch.transpose(src_path2, 0, 1), mask=mask_fre))
            src_path2 = torch.transpose(torch.transpose(src_path2, 0, 1), 0, 2)
        elif self.path == 'Time':
            src_path1 = self.activation(self.encoder_path1(
                torch.transpose(src_embedded, 0, 2), mask=mask_time))
            src_path1 = torch.transpose(src_path1, 0, 2)

            src_path2 = self.activation(self.encoder_path2(
                torch.transpose(src_embedded, 0, 2), mask=mask_time))
            src_path2 = torch.transpose(src_path2, 0, 2)
        elif self.path == 'Fre':
            src_path1 = torch.transpose(src_embedded, 0, 2)
            src_path1 = self.activation(self.encoder_path1(
                torch.transpose(src_path1, 0, 1), mask=mask_fre))
            src_path1 = torch.transpose(torch.transpose(src_path1, 0, 1), 0, 2)

            src_path2 = torch.transpose(src_embedded, 0, 2)
            src_path2 = self.activation(self.encoder_path2(
                torch.transpose(src_path2, 0, 1), mask=mask_fre))
            src_path2 = torch.transpose(torch.transpose(src_path2, 0, 1), 0, 2)
        else:
            src_path1 = 0
            src_path2 = 0
            RuntimeError('The path does not exist')

        # now src_fre is F*T*d_model

        mask = src_path1 + src_path2

        # now mask is (2*d_model)*F*T
        if self.complex_mask:
            mask = 10 * self.output_act(self.DeEmbedding(mask))
        else:
            mask = self.output_act(self.DeEmbedding(mask))
        # now mask is 1*F*T
        return mask


class Embedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.cnn1 = nn.Conv2d(2, d_model // 4, kernel_size=(3, 3), padding='same')
        self.cnn2 = nn.Conv2d(d_model // 4, d_model * 2 // 4, kernel_size=(3, 3),
                              dilation=(2, 2), padding='same')
        self.cnn3 = nn.Conv2d(d_model * 2 // 4, d_model * 3 // 4, kernel_size=(3, 3),
                              dilation=(2, 2), padding='same')
        self.cnn4 = nn.Conv2d(d_model * 3 // 4, d_model, kernel_size=(3, 3),
                              stride=(1, 2), padding=(1, 1))
        self.activation = nn.ReLU()

    def forward(self, src) -> Tensor:
        output = src.unsqueeze(0)
        return self.activation(self.cnn4(self.activation(self.cnn3(
            self.activation(self.cnn2(self.activation(self.cnn1(output)))))))).squeeze(0)


class DeEmbedding(nn.Module):
    def __init__(self, d_model, complex_mask):
        super().__init__()
        self.cnn1 = nn.ConvTranspose2d(d_model, d_model * 3 // 4, kernel_size=(3, 3),
                                       stride=(1, 2), padding=(1, 1), output_padding=(0, 1))
        # self.cnn1 = nn.Conv2d(d_model, d_model * 3 // 4, kernel_size=(3, 3),
        #                      dilation=(2, 2), padding='same')
        self.cnn2 = nn.Conv2d(d_model * 3 // 4, d_model * 2 // 4, kernel_size=(3, 3),
                              dilation=(2, 2), padding='same')
        self.cnn3 = nn.Conv2d(d_model * 2 // 4, d_model // 4, kernel_size=(3, 3),
                              dilation=(2, 2), padding='same')
        if complex_mask:
            self.cnn4 = nn.Conv2d(d_model // 4, 2, kernel_size=(3, 3), padding='same')
        else:
            self.cnn4 = nn.Conv2d(d_model // 4, 1, kernel_size=(3, 3), padding='same')
        self.activation = nn.ReLU()

    def forward(self, src) -> Tensor:
        output = src.unsqueeze(0)
        return self.cnn4(self.activation(self.cnn3(
            self.activation(self.cnn2(self.activation(self.cnn1(output))))))).squeeze(0)


class Decoder(nn.Module):
    def __init__(self, num_layers, num_channels, kernel_size, complex_mask):
        super().__init__()

        if complex_mask:
            self.conv1 = nn.Conv2d(4, 3 * num_channels, kernel_size=(1, 1), padding='same')
        else:
            self.conv1 = nn.Conv2d(3, 3 * num_channels, kernel_size=(1, 1), padding='same')
        self.relu = nn.ReLU()
        self.body = ResNetBody(num_channels, kernel_size, num_layers)
        self.conv2 = nn.Conv2d(3 * num_channels, 2, kernel_size=(1, 1), padding='same')

    def forward(self, src):
        output = src.unsqueeze(0)
        return self.conv2(self.body(self.relu(self.conv1(output)))).squeeze(0)


class ResNetBody(nn.Module):
    def __init__(self, num_channels, kernel_size, num_layers):
        super().__init__()
        each_layer = ResNetEach(num_channels, kernel_size)
        self.layers = ModuleList([copy.deepcopy(each_layer) for i in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src: Tensor) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output)
        return output


class ResNetEach(nn.Module):
    def __init__(self, num_channels, kernel_size):
        super().__init__()
        self.num_channels = num_channels
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(3 * num_channels, num_channels, kernel_size=(1, 1), padding='same')

        self.conv2_1 = nn.Conv2d(3 * num_channels, num_channels, kernel_size=(1, 1), padding='same')
        self.conv2_2 = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), padding='same')

        self.conv3_1 = nn.Conv2d(3 * num_channels, num_channels, kernel_size=(1, 1), padding='same')
        self.conv3_2 = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), padding='same')
        self.conv3_3 = nn.Conv2d(num_channels, num_channels, kernel_size=(3, 3), padding='same')

        self.conv_out = nn.Conv2d(3 * num_channels, 3 * num_channels, kernel_size=(1, 1), padding='same')

    def forward(self, src: Tensor) -> Tensor:
        tmp = src
        channel1 = self.conv1_1(tmp)
        channel2 = self.conv2_2(self.relu1(self.conv2_1(tmp)))
        channel3 = self.conv3_3(self.relu1(self.conv3_2(self.relu1(self.conv3_1(tmp)))))

        return self.relu2(self.conv_out(self.relu2(torch.cat((channel1, channel2, channel3), dim=1))) + tmp)


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, layer_norm, dropout=0.1):
        super().__init__()
        self.first_layer = First_Layer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                       layer_norm=layer_norm)
        encoder_layer = Encoder_Layer(d_model, nhead=nhead, dim_feedforward=dim_feedforward, layer_norm=layer_norm)
        self.layers = ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.dropout = Dropout(dropout)

    def forward(self,
                src: Tensor,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                is_causal: Optional[bool] = None) -> Tensor:
        if is_causal is None:
            is_causal = False
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        output = self.first_layer(src)
        for mod in self.layers:
            output = mod(output, src_mask=mask, is_causal=is_causal,
                         src_key_padding_mask=src_key_padding_mask)
        return self.dropout(output)


class Encoder_Layer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, layer_norm: bool, dropout: float = 0.0,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                             batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        # Notice, the improved Transformer turns linear1 to gru
        self.linear1 = Linear(d_model, dim_feedforward,  **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        if layer_norm:
            self.norm1 = LayerNorm(d_model, eps=layer_norm_eps,  **factory_kwargs)
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            self.norm1 = nn.Identity(**factory_kwargs)
            self.norm2 = nn.Identity(**factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = activation
        self.activation_ffc = nn.PReLU()

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                is_causal: bool = False) -> Tensor:

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src

        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x

        # self-attention block

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x, matrix = self.self_attn(x, x, x,
                                   attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=False,
                                   average_attn_weights=True,
                                   is_causal=is_causal)
        # torch.save(matrix, 'heart_time')
        return self.dropout1(x)

        # feed forward block

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation_ffc(self.linear1(x))))
        return self.dropout2(x)


class First_Layer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, layer_norm: bool, dropout: float = 0.0,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        # Notice, the improved Transformer turns linear1 to gru
        self.linear1 = nn.GRU(d_model, dim_feedforward // 2, bidirectional=True,  **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model,  **factory_kwargs)

        self.norm_first = norm_first
        if layer_norm:
            self.norm1 = LayerNorm(d_model, eps=layer_norm_eps,  **factory_kwargs)
            self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            self.norm1 = nn.Identity(**factory_kwargs)
            self.norm2 = nn.Identity(**factory_kwargs)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = activation
        self.activation_ffc = nn.PReLU()

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                is_causal: bool = False) -> Tensor:

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src

        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x

        # self-attention block

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x, matrix = self.self_attn(x, x, x,
                                   attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=False,
                                   average_attn_weights=True,
                                   is_causal=is_causal)
        # torch.save(matrix, 'heart_time')
        return self.dropout1(x)

        # feed forward block

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation_ffc(self.linear1(x)[0])))
        return self.dropout2(x)
