# 2-Dimensional Categorization

The 2-Dimensional classification involved slicing the MRI images into 2 dimensional images, and then running image classification algorithms to classify each brain slide. Obviously, this is very biologically unsound, 

## Problems

### Modularity

Many of the images are of different sizes and shapes. Currentely the traditional image processing techniques require images to be reshaped to a specific shape, but you lots alot of data. **Solved using GlobalAveraging Layers**

### Unequal Dataset

Unsure how to combat this. Feeding disproportiantely to balance the data. 

## Files
[Conv_2D.py](Conv_2D.py) First and Baseline Attempt. Implements Early Stopping & Patience. Accuracy at around 76%.  
