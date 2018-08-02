# 2-Dimensional Categorization

The 2-Dimensional classification involved slicing the MRI images into 2 dimensional images, and then running image classification algorithms to classify each brain slide. Obviously, this is very biologically unsound, 

## Problems

### Modularity

Many of the images are of different sizes and shapes. Currentely the traditional image processing techniques require images to be reshaped to a specific shape, but you lose alot of data. **Solved using GlobalAveraging Layers**

### Unequal Dataset

Unsure how to combat this. Feeding disproportiantely to balance the data. **Solved using class_weight argument in fit method** 

## Files
[Conv_2D.py](Conv_2D.py) First and Baseline Attempt. Implements Early Stopping & Patience. Accuracy at around 76%. (*After Solving Unequal Problem, Accuracy Risen to 91% with 0.05 loss*)

[VGG19.py](VGG19.py) The VGG19's model's attempt as classification. [Paper](https://arxiv.org/pdf/1409.1556.pdf) Accuracy at around 81%. 
