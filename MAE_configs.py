class MAEConfig:
    def __init__(self, encoder_embed_dim = 192, encoder_num_heads = 3, encoder_depth = 12, decoder_embed_dim = 128, decoder_num_heads = 2, decoder_depth = 4, patch_size = 16, img_size = 224, in_channels = 3):
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_num_heads = encoder_num_heads
        self.encoder_depth = encoder_depth

        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_num_heads = decoder_num_heads
        self.decoder_depth = decoder_depth

        self.patch_size = patch_size
        self.img_size = img_size
        self.in_channels = in_channels
        
    def keys(self):
        return (
            'encoder_embed_dim', 'encoder_num_heads', 'encoder_depth',
            'decoder_embed_dim', 'decoder_num_heads', 'decoder_depth',
            'patch_size', 'img_size', 'in_channels'
        )
    
    def __getitem__(self, key):
        return getattr(self, key)

# in this case, default is my default, not the default for ViT/MAE
configs = {
    'default': MAEConfig(),
    'default_with_deep_decoder': MAEConfig(decoder_depth=8),
    'default_with_weak_decoder': MAEConfig(encoder_embed_dim=256,encoder_num_heads=4, decoder_embed_dim=64, decoder_num_heads=1),
    'default_but_bigger': MAEConfig(encoder_embed_dim=384, encoder_num_heads=4, encoder_depth=12, decoder_embed_dim=128, decoder_num_heads=2, decoder_depth=4),
    'default_but_bigger_but_small_images': MAEConfig(encoder_embed_dim=256, encoder_num_heads=4, encoder_depth=12, decoder_embed_dim=192, decoder_num_heads=3, decoder_depth=8, img_size=160),
    'actually_big': MAEConfig(encoder_embed_dim=1024, encoder_num_heads=16, encoder_depth=24, decoder_embed_dim=512, decoder_num_heads=8, decoder_depth=4)
}