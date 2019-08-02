#PixelCNN

This repository is a PyTorch implementation of [PixelCNN](https://arxiv.org/abs/1601.06759) in its [gated](https://arxiv.org/abs/1606.05328) form.
The main goals I've pursued while doing so is to dive deeper into PyTorch and the network's architecture itself, which I've found both interesting and challenging to grasp. The repo might help someone, too!

A lot of ideas were taken from [rampage644](https://github.com/rampage644)'s, [blog](http://sergeiturukin.com). Useful links also include [this](https://wiki.math.uwaterloo.ca/statwiki/index.php?title=STAT946F17/Conditional_Image_Generation_with_PixelCNN_Decoders), [this](http://www.scottreed.info/files/iclr2017.pdf) and [this](https://github.com/kundan2510/pixelCNN).

#Model architecture
Here I am going to sum up the main idea behind the architecture. I won't go deep into implementation details and how convolutions work, because it would be too much text and visuals. Visit the links above in order to have a more detailed look on the inner workings of the architecture. Then come here for a summary :)

At first this architecture was an attempt to speed up the learning process of a RNN implementation of the same idea, which is a generative model that learns an explicit joint distribution of image's pixels by modeling it using simple chain rule. The order is row-wise i.e. value of each pixel depends on values of all pixels above and to the left of it. Here is an explanatory image:

![PixelCNN chain rule](https://wiki.math.uwaterloo.ca/statwiki/images/thumb/5/5b/xi_img.png/1000px-xi_img.png)

In order to achieve this property authors of the papers used simple masked convolutions, which in the case of 1-channel black and white images look like this:

![1-channel masked convolution](https://wiki.math.uwaterloo.ca/statwiki/images/thumb/f/f0/masking1.png/400px-masking1.png)

(i. e. convolutional filters are multiplied by this mask before being applied to images)

There are 2 types of masks: A and B. Masked convolution of type A can only see previously generated pixels, while mask of type B allows taking value of a pixel being predicted into consideration. Applying B-masked convolution after A-masked one preserves the causality, work it out! In the case of 3 data channels, types of masks are depicted on this image:

![Mask. types](http://sergeiturukin.com/assets/2017-02-23-195558_546x364_scrot.png)

The problem with a simple masking approach was the blind spot: when predicting some pixels, a portion of the image did not influence the prediction. This was fixed by introducing 2 separate convolutions: horizontal and vertical.  Vertical convolution performs a simple unmasked convolution and sends its outputs to a horizontal convolution, which performs a masked 1-by-N convolution. They also added gates in order to increase the predicting power of the model.

## Gated block
The main submodel of PixelCNN is a gated block, several of which are used in the network. Here is how it looks:

![Gated block](https://github.com/anordertoreclaim/PixelCNN/blob/master/.images/gated_block.png?raw=true)

## High level architecture
Here is what the whole architecture looks like:

![PixelCNN architecture](https://github.com/anordertoreclaim/PixelCNN/blob/master/.images/architecture.png?raw=true)

Causal block is the same as gated block, except that it has neither residual nor skip connections, its input is image instead of a tensor with depth of *hidden_fmaps* and it uses mask of type A instead of B of a usual gated block.
Skip results are summed and ran through a ReLu – 1x1 Conv – ReLu block. Then the final convolutional layer is applied, which outputs a tensor that represents unnormalized probabilities of each color level for each color channel of each pixel in the image.

# Training and sampling
### Train
In order to train the model, use the `python train.py` command and set optional arguments if needed.
```
$ python train.py -h
usage: train.py [-h] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                [--dataset DATASET] [--causal-ksize CAUSAL_KSIZE]
                [--hidden-ksize HIDDEN_KSIZE] [--data-channels DATA_CHANNELS]
                [--color-levels COLOR_LEVELS] [--hidden-fmaps HIDDEN_FMAPS]
                [--out-hidden-fmaps OUT_HIDDEN_FMAPS]
                [--hidden-layers HIDDEN_LAYERS] [--cuda CUDA]
                [--model-output-path MODEL_OUTPUT_PATH]
                [--samples-folder SAMPLES_FOLDER]

PixelCNN

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs to train model for
  --batch-size BATCH_SIZE
                        Number of images per mini-batch
  --dataset DATASET     Dataset to train model on. Either mnist, fashionmnist
                        or cifar.
  --causal-ksize CAUSAL_KSIZE
                        Kernel size of causal convolution
  --hidden-ksize HIDDEN_KSIZE
                        Kernel size of hidden layers convolutions
  --data-channels DATA_CHANNELS
                        Number of data channels
  --color-levels COLOR_LEVELS
                        Number of levels to quantisize value of each channel
                        of each pixel into
  --hidden-fmaps HIDDEN_FMAPS
                        Number of feature maps in hidden layer
  --out-hidden-fmaps OUT_HIDDEN_FMAPS
                        Number of feature maps in outer hidden layer
  --hidden-layers HIDDEN_LAYERS
                        Number of layers of gated convolutions with mask of
                        type "B"
  --cuda CUDA           Flag indicating whether CUDA should be used
  --model-output-path MODEL_OUTPUT_PATH, -m MODEL_OUTPUT_PATH
                        Output path for model's parameters
  --samples-folder SAMPLES_FOLDER, -o SAMPLES_FOLDER
                        Path where sampled images will be saved
```
### Sample
Sampling is performed similarly: with `python sample.py`.
```
$ python sample.py -h
usage: sample.py [-h] [--causal-ksize CAUSAL_KSIZE]
                 [--hidden-ksize HIDDEN_KSIZE] [--data-channels DATA_CHANNELS]
                 [--color-levels COLOR_LEVELS] [--hidden-fmaps HIDDEN_FMAPS]
                 [--out-hidden-fmaps OUT_HIDDEN_FMAPS]
                 [--hidden-layers HIDDEN_LAYERS] [--cuda CUDA]
                 [--model-path MODEL_PATH] [--output-fname OUTPUT_FNAME]
                 [--count COUNT] [--height HEIGHT] [--width WIDTH]

PixelCNN

optional arguments:
  -h, --help            show this help message and exit
  --causal-ksize CAUSAL_KSIZE
                        Kernel size of causal convolution
  --hidden-ksize HIDDEN_KSIZE
                        Kernel size of hidden layers convolutions
  --data-channels DATA_CHANNELS
                        Number of data channels
  --color-levels COLOR_LEVELS
                        Number of levels to quantisize value of each channel
                        of each pixel into
  --hidden-fmaps HIDDEN_FMAPS
                        Number of feature maps in hidden layer
  --out-hidden-fmaps OUT_HIDDEN_FMAPS
                        Number of feature maps in outer hidden layer
  --hidden-layers HIDDEN_LAYERS
                        Number of layers of gated convolutions with mask of
                        type "B"
  --cuda CUDA           Flag indicating whether CUDA should be used
  --model-path MODEL_PATH, -m MODEL_PATH
                        Path to model's saved parameters
  --output-fname OUTPUT_FNAME, -o OUTPUT_FNAME
                        Output filename
  --count COUNT, -c COUNT
                        Number of images to generate (is rounded to the
                        nearest integer square)
  --height HEIGHT       Output image height
  --width WIDTH         Output image width
```
