import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear_attention import Attention, crop_feature, pad_feature
from einops.einops import rearrange
from collections import OrderedDict
from ..utils.position_encoding import RoPEPositionEncodingSine
import numpy as np
from loguru import logger    
import math

class AG_RoPE_EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 agg_size0=4,
                 agg_size1=4,
                 no_flash=False,
                 rope=False,
                 npe=None,
                 fp32=False,
                 ):
        super(AG_RoPE_EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.agg_size0, self.agg_size1 = agg_size0, agg_size1
        self.rope = rope

        # aggregate and position encoding
        self.aggregate = nn.Conv2d(d_model, d_model, kernel_size=agg_size0, padding=0, stride=agg_size0, bias=False, groups=d_model) if self.agg_size0 != 1 else nn.Identity()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.agg_size1, stride=self.agg_size1) if self.agg_size1 != 1 else nn.Identity()
        if self.rope:
            self.rope_pos_enc = RoPEPositionEncodingSine(d_model, max_shape=(256, 256), npe=npe, ropefp16=True)
        
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)        
        self.attention = Attention(no_flash, self.nhead, self.dim, fp32)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.LeakyReLU(inplace = True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]
            source (torch.Tensor): [N, C, H1, W1]
            x_mask (torch.Tensor): [N, H0, W0] (optional) (L = H0*W0)
            source_mask (torch.Tensor): [N, H1, W1] (optional) (S = H1*W1)
        """
        bs, C, H0, W0 = x.size()
        H1, W1 = source.size(-2), source.size(-1)

        # Aggragate feature
        query, source = self.norm1(self.aggregate(x).permute(0,2,3,1)), self.norm1(self.max_pool(source).permute(0,2,3,1)) # [N, H, W, C]
        if x_mask is not None:
            x_mask, source_mask = map(lambda x: self.max_pool(x.float()).bool(), [x_mask, source_mask])
        query, key, value = self.q_proj(query), self.k_proj(source), self.v_proj(source)

        # Positional encoding        
        if self.rope:
            query = self.rope_pos_enc(query)
            key = self.rope_pos_enc(key)

        # multi-head attention handle padding mask
        m = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)
        m = self.merge(m.reshape(bs, -1, self.nhead*self.dim)) # [N, L, C]

        # Upsample feature
        m = rearrange(m, 'b (h w) c -> b c h w', h=H0 // self.agg_size0, w=W0 // self.agg_size0) # [N, C, H0, W0]
        if self.agg_size0 != 1:
            m = torch.nn.functional.interpolate(m, scale_factor=self.agg_size0, mode='bilinear', align_corners=False) # [N, C, H0, W0]

        # feed-forward network
        m = self.mlp(torch.cat([x, m], dim=1).permute(0, 2, 3, 1)) # [N, H0, W0, C]
        m = self.norm2(m).permute(0, 3, 1, 2) # [N, C, H0, W0]

        return x + m

class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()
        
        self.full_config = config
        self.fp32 = not (config['mp'] or config['half'])
        config = config['coarse']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        self.agg_size0, self.agg_size1 = config['agg_size0'], config['agg_size1']
        self.rope = config['rope']

        self_layer = AG_RoPE_EncoderLayer(config['d_model'], config['nhead'], config['agg_size0'], config['agg_size1'],
                                            config['no_flash'], config['rope'], config['npe'], self.fp32)
        cross_layer = AG_RoPE_EncoderLayer(config['d_model'], config['nhead'], config['agg_size0'], config['agg_size1'],
                                            config['no_flash'], False, config['npe'], self.fp32)
        self.layers = nn.ModuleList([copy.deepcopy(self_layer) if _ == 'self' else copy.deepcopy(cross_layer) for _ in self.layer_names])
        
        # --- early-stop: confidence heads ---------------------------------
        self.depth_conf = self.full_config['coarse'].get('depth_confidence', -1)
        n_layers = len(self.layers)
        print(f"LoFTR: {self.depth_conf=}, {n_layers=}")

        self.conf_heads = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(self.d_model, 1, 1), nn.Sigmoid())
            for _ in range(n_layers - 1)]        # no head for final layer
        )
        # same exponential schedule as LightGlue
        self.conf_thr = [0.8 + 0.1 * np.exp(-4.0 * i / n_layers)
                        for i in range(n_layers - 1)]

        #  initialize confidence heads
        p0 = 0.6
        logit_bias = math.log(p0 / (1 - p0))
        for head in self.conf_heads:
            nn.init.constant_(head[0].bias, p0)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None, data=None):
        """
        Args:
            feat0 (torch.Tensor): [N, C, H, W]
            feat1 (torch.Tensor): [N, C, H, W]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        H0, W0, H1, W1 = feat0.size(-2), feat0.size(-1), feat1.size(-2), feat1.size(-1)
        bs = feat0.shape[0]

        feature_cropped = False
        if bs == 1 and mask0 is not None and mask1 is not None:
            mask_H0, mask_W0, mask_H1, mask_W1 = mask0.size(-2), mask0.size(-1), mask1.size(-2), mask1.size(-1)
            mask_h0, mask_w0, mask_h1, mask_w1 = mask0[0].sum(-2)[0], mask0[0].sum(-1)[0], mask1[0].sum(-2)[0], mask1[0].sum(-1)[0]
            mask_h0, mask_w0, mask_h1, mask_w1 = mask_h0//self.agg_size0*self.agg_size0, mask_w0//self.agg_size0*self.agg_size0, mask_h1//self.agg_size1*self.agg_size1, mask_w1//self.agg_size1*self.agg_size1
            feat0 = feat0[:, :, :mask_h0, :mask_w0]
            feat1 = feat1[:, :, :mask_h1, :mask_w1]
            feature_cropped = True

        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            if feature_cropped:
                mask0, mask1 = None, None
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)                
            else:
                raise KeyError

            if data is not None and i < len(self.conf_heads):                     # skip during inference
                with torch.no_grad():             # labels, no gradient
                    # 1. compute soft match matrix *now*
                    sim_now = torch.einsum(
                        'nlc,nsc->nls',
                        feat0.flatten(2).transpose(1,2),
                        feat1.flatten(2).transpose(1,2)
                    ) / np.sqrt(feat0.size(1))
                    sim_now = F.softmax(sim_now,1) * F.softmax(sim_now,2)

                    q = data.setdefault('sim_queue', [])
                    q.append(sim_now.detach())          # push this layer
                    if len(q) >= 4:                     # have ℓ-3 and ℓ   now
                        data.setdefault('sim_pairs', [])\
                            .append((q[-4], q[-1]))     #   (ℓ-3, ℓ)
                    if len(q) > 4:                      # keep only the last 4
                        q.pop(0)

                prev = data.get('sim_prev')                 # None for layer-0
                data['sim_prev'] = sim_now.detach()         # store for next iter

                if prev is not None:                        # we have ℓ   and ℓ-1
                    data.setdefault('sim_pairs', []).append((prev, sim_now.detach()))
                data.setdefault('sim_inter', []).append(sim_now.detach())
                data.setdefault('conf_out0', []).append(self.conf_heads[i](feat0.detach()))
                data.setdefault('conf_out1', []).append(self.conf_heads[i](feat1.detach()))

            # early stop check
            # print("self.depth_conf", self.depth_conf)
            if self.depth_conf > 0 and i < len(self.layers) - 1:
                print("Early stop check at layer", i+1)
                if i == 6:
                    break
                # simple per-pixel confidence
                conf0 = self.conf_heads[i](feat0).flatten(1)   # [B, H*W]
                conf1 = self.conf_heads[i](feat1).flatten(1)
                thr = self.conf_thr[i]
                n_conf = ((conf0 >= thr).sum() + (conf1 >= thr).sum()).item()
                ratio  = n_conf / float(conf0.numel() + conf1.numel())
                # print(f"Layer {i+1}: {n_conf} / {conf0.numel() + conf1.numel()} = {ratio:.2f} > {thr:.2f}")
                print(f"Layer {i+1}: {ratio}")
                if ratio >= self.depth_conf:   # matches LightGlue’s rule
                    print(f"Early stop at layer {i+1} of {len(self.layers) - 1} with ratio {ratio:.2f} >= {self.depth_conf:.2f}")
                    break

        if self.training:
            # final reference assignment
            with torch.no_grad():
                sim_final = torch.einsum(
                    'nlc,nsc->nls',
                    feat0.flatten(2).transpose(1,2),
                    feat1.flatten(2).transpose(1,2)
                ) / np.sqrt(feat0.size(1))
                sim_final = F.softmax(sim_final,1) * F.softmax(sim_final,2)
            data['sim_final'] = sim_final.detach()

        if feature_cropped:
            # padding feature
            bs, c, mask_h0, mask_w0 = feat0.size()
            if mask_h0 != mask_H0:
                feat0 = torch.cat([feat0, torch.zeros(bs, c, mask_H0-mask_h0, mask_W0, device=feat0.device, dtype=feat0.dtype)], dim=-2)
            elif mask_w0 != mask_W0:
                feat0 = torch.cat([feat0, torch.zeros(bs, c, mask_H0, mask_W0-mask_w0, device=feat0.device, dtype=feat0.dtype)], dim=-1)

            bs, c, mask_h1, mask_w1 = feat1.size()
            if mask_h1 != mask_H1:
                feat1 = torch.cat([feat1, torch.zeros(bs, c, mask_H1-mask_h1, mask_W1, device=feat1.device, dtype=feat1.dtype)], dim=-2)
            elif mask_w1 != mask_W1:
                feat1 = torch.cat([feat1, torch.zeros(bs, c, mask_H1, mask_W1-mask_w1, device=feat1.device, dtype=feat1.dtype)], dim=-1)

        return feat0, feat1