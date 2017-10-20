# Adversarial Autoencoder
Replicates Adversarial Autoencoder architecture from [Makhzani, Alireza, et al. "Adversarial autoencoders." arXiv preprint arXiv:1511.05644 (2015)](https://arxiv.org/abs/1511.05644). 



The code is adapted from Naresh's implementation [here](https://github.com/Naresh1318/Adversarial_Autoencoder). Thanks Naresh!

## Training

To train the adversarial autoencoder, run:
```
python adversarialautoencoder.py --train
```

## Generate Sample Images

### Single Image

After training a model (at least 1 epoch), generate a single image using:
```
python adversarialautoencoder.py --sample
```

Use `-z` to input a latent vector.
For example, 
```
python adversarialautoencoder.py --sample -z [-10,-10]
```
will generate an image from latent vector [-10,-10].

### Image Grid

After training a model (at least 1 epoch), generate a grid of images using:
```
python adversarialautoencoder.py --samplegrid
```

Use `-rz1`, `-rz2`, `-nz1` and `-nz1` to specify the z1 range, z2 range, number of z1 steps and number of z2 steps.

For example,
```
python adversarialautoencoder.py --sample -rz1 [-10,10] -rz2 [-10,10] -nz1 5 -nz2 2
```
will generate 5\*2=10 images where both z1 and z2 range from -10 to 10 inclusive.

## Plot Latent Vectors

After training a model (at least 1 epoch), encode images and plot the encodings using:
```
python adversarialautoencoder.py --plot
```

Use `-i` to specify the number of images to encode and plot.

For example, 
```
python adversarialautoencoder.py --plot -i 1000
```
will encode and plot 1000 images.
