# 3-Dimensional Categorization

MRI's are typically 3-Dimensional, so biologically the correct way to classify these is using a 3-dimensional classifier. This obviously means that there is less input data, and more dimensions, but it is something I hope to combat later.
As a note, because of the ridiculous size, the GPU I'm working on CANNOT process things with a batch size of more than 1 (the resulting shape being (1, 256, 240, 176). Thus I needed to pool aggressively before applying some layers that would increase the dimensionality to a large number.

## Problems

### Modularity

Many of the images are of different sizes and shapes. Unfortunately, because many of them are ridiculously different, I decided to find the one shape that was most prominent in my dataset, and use it. It resulted in around 700 samples, with the shape being (256, 240, 176). Evidently because there are 10,813,440 dimensions, I ran into the curse of dimensionality. Will Try Some Tricks (Encoders??)

### Unequal Dataset

Unsure how to combat this. Feeding disproportionately to balance the data. **Solved using class_weight argument in fit method**

### Huge File Size

Had to set batch size to 1 to ensure GPU wouldn't crash mid-training. More specifics per file.

### GAN File Generation

Seemed to do the trick. With the augmented dataset the accuracy of the 3D Convolution model went from 91.4% to 96.6% after adding 86 more samples (increasing sample pool from 614 to 700). This was **without** the Spearmint optimization. Mission accomplished!!  

## Files
[Conv_3D.py](Conv_3D.py) First Attempt at using straight Convolutional Layers. 424 (oof) minutes per epoch of training. Fine-tuning parameters before reporting results. Accuracy: 91.4%, Loss: 0.1520

[LSTMCNN.py](LSTMCNN.py) Thought experiment... instead of using the z-dimension as a spatial dimension, make it a temporal dimension. Not necessarily biologically backed (unlike that of a fMRI), but might be useful? Uses 2 dimension convolutions after due to the LSTM reducing dimensions. As expected, metrics weren't as good as the 3D Convolution; Accuracy: 88.2%, Loss: 0.2152

[Spearmint](Spearmint) Spearmint is a Bayesian optimizer for hyperparameters or basically anything. It runs simulations to attempt to minimize a certain set of properties (Varies from number of layers, filter shape, optimizer type, etc.) More information in that folder's README.
