# Processing Files

## Files
[exportjpg.py](exportjpg.py) Because the original file format was .dcm, convert and organize image files for eventual classification/combination into 3D.

[removejpg.py](removejpg.py) Originally used to trim unessecary parts for 2D classification to be better. Eventually unused. 

[create3D.py](create3D.py) Create 3D numpy arrays based on the sizes given by the find3D file. Stores all of them into a numpy array for 3D classification.

[find3D.py](find3D.py) Unfortunately, not all the patients had the same shape for their eventual 3-D images. Instead of cropping/rescaling, I just got a dictionary of the potential sizes and their frequencies, and just chose the most common one to pass into create3D. 
