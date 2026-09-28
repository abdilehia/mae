Masked Autoencoder based on ViT-M and trained on a ImageNet-100 dataset. Trained for 400 epochs locally on a 6GB RTX2060. Used the paper 'Masked Autoencoders Are Scalable Vision Learners' and the facebookresearch code implementation as reference so both are mentioned below. Weights can be found [here](https://huggingface.co/abdilehia/mae).

Image resolution is 224x224 with a patch size of 16x16 which is pretty standard for ViT. The encoder has a width of 512 and a depth of 12 whereas the decoder has a width of 128 and depth of 4. That was about the best I could get at that resolution for my specs. MAEs already use an asymmetrical encoder and decoder so it's probably fine.

**Acknowledgements:**<br>
(Paper) He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2021). Masked Autoencoders Are Scalable Vision Learners. arXiv [Cs.CV]. Retrieved from http://arxiv.org/abs/2111.06377<br>
(Code) https://github.com/facebookresearch/mae
