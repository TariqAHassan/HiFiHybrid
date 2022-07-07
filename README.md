# HiFi Hybrid

A vocoder broadly based on [HiFi-GAN](https://arxiv.org/abs/2010.05646),
with the addition of some more recent proposals from [BigVGAN](https://arxiv.org/abs/2206.04658).

Specifically:

   * The discriminator design used is identical to HiFi, as this repository does not follow BigVGAN in replacing 
   the multi-scale discriminator (MSD) with a mutli-resolution discriminator (MSD).
     * This could be done in the future, but a preliminary investigation found little difference between the two
   * The generator design is very similar to BigVGAN in that it employs anti-aliased multi-periodicity composition (AMP)
     modules, but the low pass filters are made trainable.

Thus, this code is essentially a hybrid between HiFi and BigVGAN.

## Training

```shell
python train.py /path/to/data/goes/here
```

#### Help

Information on training the model can be found by running the following command:

```shell
$ python train.py --help

NAME
    train.py - Train Model.

SYNOPSIS
    train.py DATA_PATH <flags>

DESCRIPTION
    Train Model.

POSITIONAL ARGUMENTS
    DATA_PATH
        Type: str
        system path where audio samples exist.

FLAGS
    --file_ext=FILE_EXT
        Type: str
        Default: 'wav'
        file extension to filter for in ``data_path``.
    --val_prop=VAL_PROP
        Type: float
        Default: 0.1
        proportion of files in ``data_path`` to use for validation
    --max_epochs=MAX_EPOCHS
        Type: int
        Default: 3200
        the maximum number of epochs to train the model for

...
```

### Results

Initial results from this model are quite promising.

The BigVGAN paper leverages a lot of evaluation metrics (M-STFT, PESQ, MCD, etc.)
which, regrettably, I have not yet had time to implement. However, a simple
plot of the L1 reconstruction error over time is easy to obtain and still quite instructive.

![](/assets/mel_loss.png)

## References

  * Some code used here was adapted from https://github.com/jik876/hifi-gan
