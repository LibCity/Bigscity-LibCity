from logging import getLogger
from typing import Callable, Optional

import torch
from einops import rearrange
from einops import reduce
from einops.layers.torch import Rearrange
from torch import Tensor
from torch import nn, einsum
from torch.nn import functional as F, init

from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class Rc(nn.Module):
    def __init__(self, input_shape):
        super(Rc, self).__init__()
        self.nb_flow = input_shape[0]
        self.ilayer = iLayer(input_shape)

    def forward(self, x):
        """
            x: (*, c, h, w)
          out: (*, 2, h ,w)
        """
        # x = rearrange(x,"b (nb_flow c) h w -> b nb_flow c h w",nb_flow=self.nb_flow)
        # x = reduce(x,"b nb_flow c h w -> b nb_flow h w","sum")
        x = reduce(x, "b (c1 c) h w -> b c1 h w", "sum", c1=self.nb_flow)
        out = self.ilayer(x)
        return out


class iLayer(nn.Module):
    '''    elementwise multiplication
    '''

    def __init__(self, input_shape):
        '''
        input_shape: (,*,c,,h,w)
        self.weights shape: (,*,c,h,w)
        '''
        super(iLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(*input_shape))  # define the trainable parameter
        init.xavier_uniform_(self.weights.data)

    def forward(self, x):
        '''
        x: (batch, c, h,w)
        self.weights shape: (c,h,w)
        output: (batch, c, h,w)
        '''
        return x * self.weights


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    basicBlock
    """
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = conv1x1(inplanes, planes)
        self.convback = conv1x1(planes, inplanes)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)  # inplanes -> planes
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # planes -> planes
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # inplanes -> planes

        out += identity  # planes

        out = self.convback(out)
        out = self.relu(out)

        return out


class iLayer(nn.Module):
    '''    elementwise multiplication
    '''

    def __init__(self, input_shape):
        '''
        input_shape: (,*,c,,h,w)
        self.weights shape: (,*,c,h,w)
        '''
        super(iLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(*input_shape))  # define the trainable parameter
        # init.xavier_uniform_(self.weights.data)

    def forward(self, x):
        '''
        x: (batch, c, h,w)
        self.weights shape: (c,h,w)
        output: (batch, c, h,w)
        '''
        return x * self.weights


def pair(t):
    return t if isinstance(t, list) else [t, t]


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Softmax(nn.Module):
    def __init__(self, normlization_scale):
        super(Softmax, self).__init__()
        self.normlization_scale = normlization_scale

    def forward(self, x):
        return F.softmax(x / (self.normlization_scale ** 0.5), dim=1)


class ViT(nn.Module):

    def __init__(self, *,
                 image_size,
                 patch_size,
                 num_classes,
                 dim, depth, heads,
                 mlp_dim,
                 pool='mean',
                 channels=3,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0., seq_pool=True):
        """

        args:
        :param image_size: input map size
        :param patch_size: patch size
        :param num_classes: output size
        :param dim: embedding dimension.
        :param depth: num Of Transformer Block
        :param heads: number of head
        :param mlp_dim: dimension of MLP in transformer.
        :param pool: the way of pooling transformer output.
        :param channels: number of channels in input tensor.
        :param dim_head: embedding dimension of a head.
        :param dropout: dropout probability
        :param emb_dropout: dropout probability in transformer
        :param seq_pool: weather to use sequence pooling after transformer block.
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.dim = dim
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.num_patches = num_patches
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        mlp_dim = dim
        # if seq_pool:
        #     mlp_dim = dim * num_patches
        # else:
        #     mlp_dim = dim
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, num_classes),
            # nn.ReLU(),
            # nn.Linear(num_classes, num_classes)
        )
        self.seq_pool = seq_pool
        if seq_pool:
            self.attention_pool = nn.Linear(dim, 1)
            # self.atten_layer_norm = nn.LayerNorm([self.num_patches, dim])
            # self.attention_pool = nn.Sequential(self.atten_layer_norm,self.attention_pool)
            self.softmax_scale = dim
            self.softmax = Softmax(normlization_scale=self.softmax_scale)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n)]
        x = self.dropout(x)

        x = self.transformer(x)

        if self.seq_pool:
            x = torch.matmul(self.softmax(self.attention_pool(x)).transpose(-1, -2), x).squeeze(-2)
            # attention_sequence = self.softmax(self.attention_pool(x))
            # attention_sequence = repeat(attention_sequence, "b n d -> b n (d dim)", dim=self.dim)
            # x = x * attention_sequence
            # x = rearrange(x, "b n d -> b (n d)")


        else:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)

        return x


class STTSNet(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        """

        """
        super().__init__(config, data_feature)

        self._logger = getLogger()

        # data feature
        self.load_external = data_feature.get('load_external', False)

        # parse model config
        map_height = config.get('map_height', 32)
        map_width = config.get('map_width', 32)
        patch_size = config.get('patch_size', 4)
        close_channels = config.get('close_channels', 6)
        trend_channels = config.get('trend_channels', 6)
        close_dim = config.get('close_dim', 1024)
        trend_dim = config.get('trend_dim', 1024)
        close_depth = config.get('close_depth', 4)
        trend_depth = config.get('trend_depth', 4)
        close_head = config.get('close_head', 2)
        trend_head = config.get('trend_head', 2)
        close_mlp_dim = config.get('close_mlp_dim', 2048)
        trend_mlp_dim = config.get('trend_mlp_dim', 2048)
        nb_flow = config.get('nb_flow', 2)
        seq_pool = config.get('seq_pool', True)
        pre_conv = config.get('pre_conv', True)
        shortcut = config.get('shortcut', True)
        conv_channels = config.get('conv_channels', 64)
        drop_prob = config.get('drop_prob', 0.1)
        conv3d = config.get('conv3d', False)

        # model config
        self.map_height = map_height
        self.map_width = map_width
        self.nb_flow = nb_flow
        output_dim = nb_flow * map_height * map_width
        close_dim_head = int(close_dim / close_head)
        trend_dim_head = int(trend_dim / close_head)

        self.pre_conv = pre_conv
        self.conv3d = conv3d
        if pre_conv:
            self.pre_close_conv = nn.Sequential(
                BasicBlock(inplanes=close_channels, planes=conv_channels),
                # BasicBlock(inplanes=close_channels,planes=conv_channels),
            )
            self.pre_trend_conv = nn.Sequential(
                BasicBlock(inplanes=trend_channels, planes=conv_channels),
                # BasicBlock(inplanes=trend_channels,planes=conv_channels)
            )

        # close_channels, trend_channels = nb_flow * close_channels, nb_flow * trend_channels

        self.closeness_transformer = ViT(
            image_size=[map_height, map_width],
            patch_size=patch_size,
            num_classes=output_dim,
            dim=close_dim,
            depth=close_depth,
            heads=close_head,
            mlp_dim=close_mlp_dim,
            dropout=drop_prob,
            emb_dropout=drop_prob,
            channels=close_channels,
            dim_head=close_dim_head,
            seq_pool=seq_pool
        )
        self.trend_transformer = ViT(
            image_size=[map_height, map_width],
            patch_size=patch_size,
            num_classes=output_dim,
            dim=trend_dim,
            depth=trend_depth,
            heads=trend_head,
            mlp_dim=trend_mlp_dim,
            dropout=drop_prob,
            emb_dropout=drop_prob,
            channels=trend_channels,
            dim_head=trend_dim_head,
            seq_pool=seq_pool,

        )
        input_shape = (nb_flow, map_height, map_width)

        self.shortcut = shortcut
        if shortcut:
            self.Rc_Xc = Rc(input_shape)
            self.Rc_Xt = Rc(input_shape)
            # self.Rc_conv_Xc = Rc(input_shape)
            # self.Rc_conv_Xt = Rc(input_shape)

        self.close_ilayer = iLayer(input_shape=input_shape)
        self.trend_ilayer = iLayer(input_shape=input_shape)

    def forward(self, xc, xt, x_ext=None):
        """

        :param xc: batch size, num_close,map_height,map_width
        :param xt: batch size, num_week,map_height,map_width
        :return:
        """
        if len(xc.shape) == 5:
            # reshape 5 dimensions to 4 dimensions.
            xc, xt = list(map(lambda x: rearrange(x, "b n l h w -> b (n l) h w"), [xc, xt]))
        batch_size = xc.shape[0]
        identity_xc, identity_xt = xc, xt
        if self.pre_conv:
            xc = self.pre_close_conv(xc)
            xt = self.pre_trend_conv(xt)

        close_out = self.closeness_transformer(xc)
        trend_out = self.trend_transformer(xt)

        # relu + linear

        close_out = close_out.reshape(batch_size, self.nb_flow, self.map_height, self.map_width)
        trend_out = trend_out.reshape(batch_size, self.nb_flow, self.map_height, self.map_width)

        close_out = self.close_ilayer(close_out)
        trend_out = self.trend_ilayer(trend_out)
        out = close_out + trend_out

        if self.shortcut:
            shortcut_out = self.Rc_Xc(identity_xc) + self.Rc_Xt(identity_xt)
            # +self.Rc_conv_Xc(xc_conv)+self.Rc_conv_Xt(xt_conv)
            out += shortcut_out

        if not self.training:
            out = out.relu()

        return out

    def predict(self, batch):
        if self.load_external:
            xc, xt, x_ext = batch['xc'], batch['xt'], batch['x_ext']
        else:
            xc, xt, x_ext = batch['xc'], batch['xt'], None
        return self.forward(xc, xt, x_ext)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        return loss.masked_mse_torch(y_predicted, y_true)
