# Adversarial Autoencoder
Replicates Adversarial Autoencoder architecture from [Makhzani, Alireza, et al. "Adversarial autoencoders." arXiv preprint arXiv:1511.05644 (2015)](https://arxiv.org/abs/1511.05644). 

![alt text](https://raw.githubusercontent.com/greentfrapp/adversarialautoencoder/master/basic_architecture.png "Basic Adversarial Autoencoder Architecture")

The code is adapted from Naresh's implementation [here](https://github.com/Naresh1318/Adversarial_Autoencoder). Thanks Naresh!

## General

For help and details on options, just run the script without any options:
```
python adversarialautoencoder.py
```

## Training

To train the adversarial autoencoder, run:
```
python adversarialautoencoder.py --train
```

This downloads the MNIST dataset into the `./Data` directory.
It also creates a `./Results` directory to store log files, Tensorboard files and saved models.

## Generate Sample Images

*Note! If the `--z_dim` flag was used during training, the same `--z_dim` should be specified when generating images.*

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

Image Grid after training for 1000 epochs, batch size 100, learning rate 0.001 and beta1 0.9 on AdamOptimizer, both z1 and z2 range from -10 to 10 inclusive with 10 steps each.

![alt text](https://raw.githubusercontent.com/greentfrapp/adversarialautoencoder/master/samplegrid.png "Sample Image Grid")

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

Plot of latent vectors after training for 1000 epochs, batch size 100, learning rate 0.001 and beta1 0.9 on AdamOptimizer, with 10000 images from the MNIST test set and z_dim = 2.

![alt text](https://raw.githubusercontent.com/greentfrapp/adversarialautoencoder/master/plot.png "Latent Space Plot")

The images map poorly to the latent space and as mentioned in Makhanzi (2015) Appendix A.1, the dimension of the latent space should match the intrinsic dimensionality of the data (5 to 8 in the case of MNIST).

Here is a plot of the latent vectors with z_dim = 8, visualized via t-SNE mapping. Notice the more well-defined clusters across the different classes.

![alt text](https://raw.githubusercontent.com/greentfrapp/adversarialautoencoder/master/plot2.png "Latent Space Plot")

