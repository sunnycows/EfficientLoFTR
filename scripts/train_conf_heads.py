# train_conf_heads.py

import torch, cv2, glob, os
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from src.loftr.loftr import LoFTR
from src.utils.misc import lower_config
from torch.nn import functional as F
import pandas as pd

# PAIR_ROOT   = '../_Experiments/clean_patches'
PAIR_ROOT = '../../Data/AirSim/Ch1'
CKPT_PATH   = 'weights/eloftr_outdoor.ckpt'

class PairDataset(Dataset):
    def __init__(self, root, patA='*_drone640.png', patB='*_sat640.png', size=640):
        print('Loading dataset from:', root, 'from:', os.getcwd())
        self.A = sorted(glob.glob(os.path.join(root, patA)))
        self.B = sorted(glob.glob(os.path.join(root, patB)))
        assert len(self.A) == len(self.B)
        self.HW = size

    def __len__(self):  return len(self.A)

    def _load(self, p):
        img = cv2.imread(p, 0)
        img = cv2.resize(img, (self.HW, self.HW), cv2.INTER_AREA)
        return torch.tensor(img/255., dtype=torch.float32)[None]  # [1,H,W]

    def __getitem__(self, i):
        return {'image0': self._load(self.A[i]),
                'image1': self._load(self.B[i]),
                'spv_a_ids': torch.empty(0, dtype=torch.long),
                'spv_b_ids': torch.empty(0, dtype=torch.long)}


class PairDataModule(pl.LightningDataModule):
    def __init__(self, root, bs=4, nw=4):
        super().__init__()
        self.root, self.bs, self.nw = root, bs, nw
    def setup(self, stage=None):
        full = PairDataset(self.root)
        n = len(full); n_tr = int(0.9*n)
        self.train, self.val = random_split(full, [n_tr, n-n_tr])
    def train_dataloader(self):
        return DataLoader(self.train, self.bs, True,  num_workers=self.nw,
                          pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.val,   self.bs, False, num_workers=self.nw,
                          pin_memory=True)

# ---------------------------------------------------------------

class AirSimDataset(Dataset):
    def __init__(
        self,
        root: str,
        suffix: str = 'val',
        size: int = 640
    ):
        self.root = root
        self.HW = size

        df = pd.read_csv(os.path.join(root, 'pair_csv', 'pair_list_' + suffix + '.csv'))
        df = df[df['label'] == 1]
        self.A = [os.path.join(root, p) for p in df['image0'].tolist()]
        self.B = [os.path.join(root, p) for p in df['image1'].tolist()]

        assert len(self.A) == len(self.B), \
            f"Found {len(self.A)} drone vs {len(self.B)} sat images"

    def __len__(self):
        return len(self.A)

    def _load(self, p: str):
        img = cv2.imread(p, 0)
        img = cv2.resize(img, (self.HW, self.HW), cv2.INTER_AREA)
        return torch.tensor(img / 255., dtype=torch.float32)[None]  # [1,H,W]

    def __getitem__(self, i: int):
        return {
            'image0':     self._load(self.A[i]),
            'image1':     self._load(self.B[i]),
            'spv_a_ids':  torch.empty(0, dtype=torch.long),
            'spv_b_ids':  torch.empty(0, dtype=torch.long),
        }


class AirSimDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        bs: int = 4,
        nw: int = 4
    ):
        super().__init__()
        self.root       = root
        self.batch_size = bs
        self.num_workers= nw

    def setup(self, stage=None):
        # instantiate two separate datasets from two CSVs
        self.train = AirSimDataset(self.root, suffix='train')
        self.val   = AirSimDataset(self.root)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


# ---------- Loftr model with only conf_heads trainable ----------
from src.config.default import get_cfg_defaults
from src.lightning.lightning_loftr import PL_LoFTR

class PL_LoFTRConf(PL_LoFTR):
    def __init__(self, cfg, ckpt):
        super().__init__(cfg)
        # load checkpoint (ignore new params)
        state = torch.load(ckpt, map_location='cpu')
        self.matcher.load_state_dict(state['state_dict'], strict=False)

        # freeze everything first
        for p in self.matcher.parameters():
            p.requires_grad_(False)
        # un-freeze confidence heads
        for head in self.matcher.loftr_coarse.conf_heads:
            for p in head.parameters():
                p.requires_grad_(True)

    # optimiser over trainable params only
    def configure_optimizers(self):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=3e-4, weight_decay=0.0)

class ConfHeadTrainer(pl.LightningModule):

    def __init__(self, cfg, ckpt):
        super().__init__()
        self.save_hyperparameters(ignore=['cfg'])   # for checkpoint meta
        self.cfg = cfg

        # matcher
        self.matcher = LoFTR(config=cfg['loftr'])
        self.matcher.coarse_matching.train(False)

        state = torch.load(ckpt, map_location='cpu')
        self.matcher.load_state_dict(state['state_dict'], strict=False)

        # freeze backbone
        for p in self.matcher.parameters():
            p.requires_grad_(False)
        for head in self.matcher.loftr_coarse.conf_heads:
            for p in head.parameters():
                p.requires_grad_(True)
    
    def forward(self, batch):
        # run LoFTR; it writes results into `batch` and returns None
        _ = self.matcher(batch)
        return batch  

    # def on_after_backward(self):        # <-- new hook
    #     # print one scalar per conf-head weight tensor
    #     for n, p in self.named_parameters():
    #         if 'conf_heads' in n and p.grad is not None:
    #             mean_grad = p.grad.abs().mean().item()
    #             print(f'{n}: {mean_grad:.3e}')
    #     # optional: print a blank line to separate batches
    #     print('─' * 40)

    def training_step(self, batch, _):
        self.matcher.coarse_matching.eval()        # ← disables padding branch
        self.matcher.loftr_coarse.train()        #  ← add this line
        self.matcher.fine_matching.eval()        #  ←  add this line
        out = self(batch)                    # forward pass

        loss = self._conf_loss(out)
        self.log('train/loss_conf', loss)
        return loss

    def validation_step(self, batch, _):
        out = self(batch)
        loss = self._conf_loss(out)
        self.log('val/loss_conf', loss, prog_bar=True)

    # --------------------------------------------------------------
    def confidence_bce(self, conf0, conf1, sim_now, sim_final):
        """
        BCE loss that teaches conf_heads to say “my match is final”.
        Args
            conf0, conf1 : [B, N, 1] logits from confidence heads (detach OK)
            sim_now      : [B, N, M] log-probs (or softmax) at current layer
            sim_final    : [B, N, M] reference from the *last* layer
        """
        # best partner index for every point
        idx_row_now  = sim_now.argmax(-1)          # [B, N]
        idx_row_fin  = sim_final.argmax(-1)
        idx_col_now  = sim_now.argmax(-2)
        idx_col_fin  = sim_final.argmax(-2)

        lbl0 = (idx_row_now == idx_row_fin).float()   # [B, N]
        lbl1 = (idx_col_now == idx_col_fin).float()

        # ----- reshape logits to [B, N] --------------------------------
        logit0 = conf0.flatten(2).squeeze(1)      # [B, N]
        logit1 = conf1.flatten(2).squeeze(1)      # [B, N]

        bce = F.binary_cross_entropy_with_logits
        return 0.5 * (bce(logit0, lbl0) + bce(logit1, lbl1))

    def _conf_loss(self, out):
        loss = 0.0
        for c0, c1, sim_now in zip(out['conf_out0'],
                                   out['conf_out1'],
                                   out['sim_inter']):
            sim_final = out.get('sim_final', sim_now.detach())   # fall-back if key is absent
            loss += self.confidence_bce(c0, c1, sim_now, sim_final)
        return loss / max(1, len(out['sim_inter']))

    # --------------------------------------------------------------
    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.AdamW(params, lr=3e-3, weight_decay=0.)


cfg = get_cfg_defaults()
cfg.defrost()
cfg.LOFTR.BACKBONE_TYPE           = 'RepVGG'   # what the repo uses
cfg.LOFTR.BACKBONE.VERSION        = 'A0'
cfg.LOFTR.COARSE.NPE              = 16
cfg.LOFTR.D_MODEL                 = 256
cfg.LOFTR.COARSE.NHEAD            = 8
cfg.LOFTR.SPARSE_SPVS = False      # <- add this before cfg.freeze()

# turn OFF other losses, leave only confidence loss
cfg.LOFTR.LOSS.COARSE_WEIGHT = 0.0
cfg.LOFTR.LOSS.FINE_WEIGHT   = 0.0
cfg.LOFTR.LOSS.LOCAL_WEIGHT  = 0.0
cfg.LOFTR.LOSS.CONF_WEIGHT   = 0.1     # you added this key earlier

# run all layers while training (labels need final layer)
cfg.LOFTR.COARSE.DEPTH_CONFIDENCE = -1

cfg.LOFTR.COARSE.TRAIN_COARSE_PERCENT = 0.0
cfg.LOFTR.COARSE.TRAIN_PAD_NUM_GT_MIN = 0

cfg.LOFTR.COARSE.NPE = [640, 640, 640, 640]  # training at 640 resolution
cfg.freeze()

dm     = AirSimDataModule(PAIR_ROOT, bs=1)
model  = ConfHeadTrainer(lower_config(cfg), CKPT_PATH)

trainer = pl.Trainer(
    max_epochs=2, gpus=1, precision=16,
    accumulate_grad_batches=4,            # keeps LR steady
    num_sanity_val_steps=0,
    default_root_dir='logs/conf_heads'
)
trainer.fit(model, datamodule=dm)