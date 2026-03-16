import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, width, height, in_channels, emb_dim, patch_size=16):
        super().__init__()
        assert width == height
        self.img_size = (width, height)
        self.img_size_flat = width * height
        self.in_channels = in_channels
        self.num_patches = self.img_size_flat // (patch_size ** 2)
        self.patch_size = patch_size
        
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=emb_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(start_dim=2)
    def forward(self, x):
        _, _, H, W = x.shape
        assert H == self.img_size[0]
        assert W == self.img_size[1]

        x = self.proj(x)
        x = self.flatten(x).transpose(1, 2)
        return x

if __name__ == "__main__":
    from torchvision.io import read_image, ImageReadMode
    from torchvision.transforms import v2

    patch_embed = PatchEmbedding(224, 224, 3, 512)

    data_augmentation = v2.Compose([
        v2.ToImage(),
        v2.Resize(224),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
    ])

    tensors = []

    images = [
        "image.png"
    ]

    for image in images:
        img = read_image(image, mode=ImageReadMode.RGB)
        tensors.append(data_augmentation(img))

    
    tensors = torch.stack(tensors, dim=0)

    print(f"Before patching. Shape: {tensors.shape}")

    patches = patch_embed(tensors)

    print(f"After patching. Shape {patches.shape}")