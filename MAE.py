from transformer_blocks import ViTTransformerBlock
from patch_embed import PatchEmbedding
from positional_embed import get_2d_sincos_pos_embed
import torch
import torch.nn as nn
import numpy as np

# Code in this folder is a combination of my own code as well as code from these sources:
# https://medium.com/@ovularslan/masked-autoencoders-mae-the-art-of-seeing-more-by-masking-most-pytorch-implementation-4566e08c66a6
# https://github.com/facebookresearch/mae
# https://github.com/huggingface/pytorch-image-models

# Also used this paper to learn about masked autoencoders on a theory level before starting on this
# https://arxiv.org/abs/2111.06377 (Masked Autoencoders Are Scalable Vision Learners)

# I tend to write stuff out and try to understand it so might not be an exact rip
# This file is like barely my code if any at all (bro thinks he's part of the team)
# I definitely at least organised the code and used a mix of approaches between the sources linked above
# So any slight differences are just me putting the code in terms I understand
# Not trying to cover anything up or claim its mine because I read it, googled some stuff then changed some variable names or added some comments
# The notes are aimed at me so any parts without notes are just things I already know or think are self-explanatory

class MAE(nn.Module):
    def __init__(self, encoder_embed_dim, encoder_num_heads, encoder_depth, decoder_embed_dim, decoder_num_heads, decoder_depth, patch_size, img_size, in_channels):
        super().__init__()
        print(img_size)

        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_num_heads = encoder_num_heads
        self.encoder_depth = encoder_depth

        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_num_heads = decoder_num_heads
        self.decoder_depth = decoder_depth

        self.patch_size = patch_size
        self.img_size = img_size
        self.in_channels = in_channels

        self.patch_embed = PatchEmbedding(
                                        self.img_size, 
                                        self.img_size, 
                                        self.in_channels, 
                                        self.encoder_embed_dim, 
                                        self.patch_size
                                        )
        

        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))
        self.register_buffer('encoder_pos_embed', get_2d_sincos_pos_embed(self.encoder_embed_dim, int(self.patch_embed.num_patches**.5), cls_token=True))
        
        self.encoder_blocks = nn.ModuleList([
            ViTTransformerBlock(self.encoder_embed_dim, self.encoder_num_heads) for _ in range(self.encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(self.encoder_embed_dim)

        self.decoder_embed = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.register_buffer('decoder_pos_embed', get_2d_sincos_pos_embed(self.decoder_embed_dim, int(self.patch_embed.num_patches**.5), cls_token=True))
        
        self.decoder_blocks = nn.ModuleList([
            ViTTransformerBlock(self.decoder_embed_dim, self.decoder_num_heads) for _ in range(self.decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(self.decoder_embed_dim)
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, (self.patch_size ** 2) * self.in_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Initialising learned parameters using normal distribution (values near the center are more likely than extremes)
        # 0.02 is just a number from original ViT and BERT papers
        if m is self:
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.mask_token, std=0.02)

        # Xavier Uniform or Glorot Initialisation
        # It is basically a uniform distribution (entirely random) within ranges based on the number of input and 
        # output neurons.
        if isinstance(m, nn.Conv2d): # Conv2D is treated same as linear layers
            patcher_weights = self.patch_embed.proj.weight.data
            nn.init.xavier_uniform_(patcher_weights.view([patcher_weights.shape[0], -1]))

        # Same thing with the linear layers
        # Ensures variance of inputs is the same as variance for outputs
        # Initializing bias as 0 to encourage weights rather than compensating with bias
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio)) # Number of visible patches to keep

        noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        # torch.gather is essentially using one tensor as indices to gather values in another tensor
        # so we use torch.gather to get the values at ids_keep from x
        # the unsqueeze and repeat are because the index tensor must have the same number of dimensions as the input tensor
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Create binary mask to track which patches were masked out
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore) # Re-orders patches back to original order but now we can use binary to track which ones were masked

        return x_masked, mask, ids_restore
    
    def patchify(self, imgs: torch.Tensor):
        B, C, H, W = imgs.shape

        h = H // self.patch_size
        w = W // self.patch_size
        p = self.patch_size

        # Patchify target images
        imgs = imgs.reshape(B, C, h, p, w, p)
        imgs = imgs.permute(0, 2, 4, 3, 5, 1) # (B, h, w, p, p, C)
        imgs = imgs.reshape(-1, h*w, p*p*C) # (B, hw, ppC)
        # If H and W are 224, patch_size is 16 and C is 3
        # h and w are 14 and p is 16
        # The final shape is (B, 196, 768)

        return imgs
    
    def unpatchify(self, imgs: torch.Tensor):
        # See patchify to make any sense of this
        # Just tried to reverse using known values
        p = self.patch_size
        C = self.in_channels
        h = w = self.img_size // p

        imgs = imgs.reshape(-1, h, w, p, p, C)
        imgs = imgs.permute(0, 5, 1, 3, 2, 4)
        imgs = imgs.reshape(-1, C, h*p, w*p)

        return imgs
    
    def encode(self, x: torch.Tensor, mask_ratio: float):
        x = self.patch_embed(x) # Embed patches
        
        x = x + self.encoder_pos_embed[:, 1:, :] # Add position embedding without cls_token
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Append cls_token
        cls_token = self.cls_token + self.encoder_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply transformer blocks
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore
    
    def decode(self, x, ids_restore):
        x = self.decoder_embed(x)

        # Append mask token (very little clue about the specifics here)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1) # No cls_token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) # Unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1) # Append cls_token

        x = x + self.decoder_pos_embed # Add position embeddding

        # Apply transformer blocks
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        # Remove cls_token
        x = x[:, 1:, :]

        return x

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.encode(imgs, mask_ratio)
        pred = self.decode(latent, ids_restore)
        loss = self.calculate_loss(imgs, pred, mask)
        return loss, pred, mask

    
    def calculate_loss(self, imgs: torch.Tensor, pred, mask):
        # Patchify the target images
        imgs = self.patchify(imgs)
        
        # Normalize target images (paper talks about this giving better results)
        mean = imgs.mean(dim=-1, keepdim=True)
        var = imgs.var(dim=-1, keepdim=True)
        imgs = (imgs - mean) / (var + 1.e-6)**.5 # Epsilon value to avoid dividing by zero

        # Calculate MSE on only the masked parts
        # Essentially, since the shapes don't match up, we can't just calculate MSE directly
        # So we calculate mean per patch and then get the mean of that
        loss = (imgs - pred) ** 2
        loss = torch.mean(loss, dim=-1) # Mean loss per patch
        loss = (loss * mask).sum() / mask.sum() # Mean loss on all masked patches

        return loss


from torch.utils.data import DataLoader, Dataset
import os

from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2

class MAEDataset(Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __getitem__(self, idx):
        img_tensor = read_image(self.file_list[idx], mode=ImageReadMode.RGB)
        images = self.transform(img_tensor)
        return images

    def __len__(self):
        return len(self.file_list)

BASE_DIR = "./data/COCO" # Would make no sense to leave my data path in so do what you want here
files = [os.path.join(BASE_DIR, file) for file in os.listdir(BASE_DIR) if file.endswith(".png") or file.endswith(".jpg")]
files = np.array(files[:-1], dtype=str)

augmentation = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(224, scale=(0.2, 1.0)),
    v2.ToDtype(torch.float32, scale=True),
])

data = MAEDataset(files, augmentation)
train_loader = DataLoader(data, batch_size=128, num_workers=4, prefetch_factor=4, pin_memory=True, shuffle=True, drop_last=True)


if __name__ == "__main__":
    import torch.optim as optim
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    from MAE_configs import configs

    import os
    ckpt_path = "./checkpoints"
    os.makedirs(ckpt_path, exist_ok=True)


    import torch.multiprocessing as mp
    try:
         mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MAE(**configs["default"]).train().to(device)


    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = optim.AdamW([
        {'params': decay, 'weight_decay': 0.05},
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=6e-4, betas=(0.9, 0.95))


    total_epochs = 400
    warmup_epochs = 40
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)

    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )


    epoch = 0
    # checkpoint = torch.load(os.path.join(ckpt_path, 'MAE_400.pth'), map_location=device, weights_only=False)
    # model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    # epoch = checkpoint["epoch"] + 1


    accumulation_steps = 8
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epoch, total_epochs):
        print(f"Epoch {epoch}")
        total_train_loss = 0
        
        for batch_idx, images in enumerate(train_loader):
            loss, pred, mask = model(images.to(device))
            
            loss_val = loss.item()
            loss  = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()    
                optimizer.zero_grad(set_to_none=True)

            total_train_loss += loss_val

            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}. Loss={loss_val:.4f}")

        scheduler.step()
        print(f"Avg Loss = {total_train_loss / len(train_loader):.4f}.")

        if epoch % 10 == 0:
            print(f"Saving checkpoint for epoch {epoch}/{total_epochs}", end="\n\n")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(ckpt_path, f"MAE_{epoch}.pth"))
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, os.path.join(ckpt_path, f"MAE_{total_epochs}.pth"))
    torch.save(model.state_dict(), os.path.join(ckpt_path, "MAE.pth"))

    print("Training finished. Checkpoints and final weights saved to checkpoints folder.")