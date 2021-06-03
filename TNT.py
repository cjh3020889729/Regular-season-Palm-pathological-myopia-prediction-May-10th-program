import paddle
from paddle import nn

import math
import numpy as np

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 2, 'input_size': (3, 600, 600), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'pixel_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'tnt_s_patch16_224': _cfg(
        url='',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_b_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class Identity(nn.Layer):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])
    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + paddle.rand(shape=shape, dtype=x.dtype, device=x.device)
    random_tensor.floor()  # binarize
    output = x.divide(keep_prob) * random_tensor

    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Layer):
    '''
        注意力部分
    '''

    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qk = nn.Linear(dim, hidden_dim * 2, bias_attr=qkv_bias)
        self.v = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # no inplace
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, inputs):
        x = inputs
        B, N, C = x.shape
        qk = self.qk(x).reshape((B, N, 2, self.num_heads, self.head_dim)).transpose((2, 0, 3, 1, 4))
        q, k = qk[0], qk[1]
        v = self.v(x).reshape((B, N, self.num_heads, -1)).transpose((0, 2, 1, 3))

        attn = paddle.matmul(q, k.transpose((0, 1, 3, 2))) * self.scale
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = paddle.matmul(attn, v).transpose((0, 2, 1, 3)).reshape((B, N, -1))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Layer):
    """ TNT Block
    """
    def __init__(self, dim, in_dim, num_pixel, num_heads=12, in_num_head=4, mlp_ratio=4.,
            qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        # Inner transformer
        self.norm_in = norm_layer(in_dim)
        self.attn_in = Attention(
            in_dim, in_dim, num_heads=in_num_head, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        
        self.norm_mlp_in = norm_layer(in_dim)
        self.mlp_in = Mlp(in_features=in_dim, hidden_features=int(in_dim * 4),
            out_features=in_dim, act_layer=act_layer, drop=drop)
        
        self.norm1_proj = norm_layer(in_dim)
        self.proj = nn.Linear(in_dim * num_pixel, dim, bias_attr=True)
        # Outer transformer
        self.norm_out = norm_layer(dim)
        self.attn_out = Attention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        
        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
            out_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, pixel_embed, patch_embed):
        # inner
        pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        # outer
        B, N, C = patch_embed.shape
        patch_embed[:, 1:] = patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape((B, N - 1, -1)))
        patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
        patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        return pixel_embed, patch_embed


class PixelEmbed(nn.Layer):
    """ Image to Pixel Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, in_dim=48, stride=4):
        super(PixelEmbed, self).__init__()
        num_patches = (img_size // patch_size) ** 2
        self.img_size = img_size
        self.num_patches = num_patches
        self.in_dim = in_dim
        new_patch_size = math.ceil(patch_size / stride)
        self.new_patch_size = new_patch_size

        self.proj = nn.Conv2D(in_chans, self.in_dim, kernel_size=7, padding=3, stride=stride)

    def forward(self, x, pixel_pos):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.proj(x)
        x = nn.functional.unfold(x=x, kernel_sizes=self.new_patch_size, strides=self.new_patch_size)

        x = x.transpose((0, 2, 1)).reshape((B * self.num_patches, self.in_dim, self.new_patch_size, self.new_patch_size))
        x = x + pixel_pos

        x = x.reshape((B * self.num_patches, self.in_dim, -1)).transpose((0, 2, 1))
        return x


class TNT(nn.Layer):
    """ Transformer in Transformer - https://arxiv.org/abs/2103.00112
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, in_dim=48, depth=12,
                 num_heads=12, in_num_head=4, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, first_stride=4):
        super(TNT, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.pixel_embed = PixelEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, in_dim=in_dim, stride=first_stride)
        num_patches = self.pixel_embed.num_patches
        self.num_patches = num_patches
        new_patch_size = self.pixel_embed.new_patch_size
        num_pixel = new_patch_size ** 2
        
        self.norm1_proj = norm_layer(num_pixel * in_dim)
        self.proj = nn.Linear(num_pixel * in_dim, embed_dim)
        self.norm2_proj = norm_layer(embed_dim)
            
        # 创建参数
        self.cls_token = paddle.create_parameter((1, 1, embed_dim), 'float32', attr=nn.initializer.Assign(paddle.zeros((1, 1, embed_dim))))
        self.patch_pos = paddle.create_parameter((1, num_patches + 1, embed_dim), 'float32', attr=nn.initializer.Assign(paddle.zeros((1, num_patches + 1, embed_dim))))
        self.pixel_pos = paddle.create_parameter((1, in_dim, new_patch_size, new_patch_size), 'float32', attr=nn.initializer.Assign(paddle.zeros((1, in_dim, new_patch_size, new_patch_size))))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x for x in paddle.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        for i in range(depth):
            blocks.append(Block(
                dim=embed_dim, in_dim=in_dim, num_pixel=num_pixel, num_heads=num_heads, in_num_head=in_num_head,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer))
        self.blocks = nn.LayerList(blocks)
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        with paddle.no_grad():
            self.cls_token = paddle.create_parameter(self.cls_token.shape, 'float32', attr=nn.initializer.Assign(paddle.normal(self.cls_token, std=.02)))
            self.patch_pos = paddle.create_parameter(self.patch_pos.shape, 'float32', attr=nn.initializer.Assign(paddle.normal(self.patch_pos, std=.02)))
            self.pixel_pos = paddle.create_parameter(self.pixel_pos.shape, 'float32', attr=nn.initializer.Assign(paddle.normal(self.pixel_pos, std=.02)))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with paddle.no_grad():
                m.weight = paddle.create_parameter(m.weight.shape, 'float32', attr=nn.initializer.Assign(paddle.normal(m.weight, std=.02)))
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias = paddle.create_parameter(m.bias.shape, 'float32', attr=nn.initializer.Constant(value=0.))
        elif isinstance(m, nn.LayerNorm):
            m.bias = paddle.create_parameter(m.bias.shape, 'float32', attr=nn.initializer.Constant(value=0.))
            m.weight = paddle.create_parameter(m.weight.shape, 'float32', attr=nn.initializer.Constant(value=1.))

    def no_weight_decay(self):
        return {'patch_pos', 'pixel_pos', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        pixel_embed = self.pixel_embed(x, self.pixel_pos)
        
        patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape((B, self.num_patches, -1)))))
        patch_embed = paddle.concat((self.cls_token.expand([B, self.cls_token.shape[1],self.cls_token.shape[2]]), patch_embed), axis=1)  # expand
        patch_embed = patch_embed + self.patch_pos
        patch_embed = self.pos_drop(patch_embed)

        for blk in self.blocks:
            pixel_embed, patch_embed = blk(pixel_embed, patch_embed)

        patch_embed = self.norm(patch_embed)
        return patch_embed[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def tnt_s_patch16_224(pretrained=False, **kwargs):
    model = TNT(patch_size=16, embed_dim=384, in_dim=24, depth=12, num_heads=6, in_num_head=4,
        qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_s_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


def tnt_b_patch16_224(pretrained=False, **kwargs):
    model = TNT(patch_size=16, embed_dim=640, in_dim=40, depth=12, num_heads=10, in_num_head=4,
        qkv_bias=False, **kwargs)
    model.default_cfg = default_cfgs['tnt_b_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model