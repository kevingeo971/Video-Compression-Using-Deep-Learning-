

# Welcome to the Eva Storage Module

### The storage module provides end to end help reducing the input datasize for fast video analytics.


#### Functionality within this module:
1. Using a pre-trained model on the UE-DETRAC Dataset to get compressed clustered representations.
2. Training the model on your own data to get a latent representation optimized for your own input video.


#### Primary Outputs:
- `_.txt` file with the indexes of the representative frames from your input dataset. (Can use UE-DETRAC)
- `_.avi` compressed video in mp4 version. Can be used to extract representative frames within the video dataset.


## Usage:
1. If using UE-DETRAC dataset user can skip directly to the main function.
2. Using your own individual video, the video likely needs to be preprocesed and files need to be labelled in sequential order for input to train function.

### preprocess_data.py
Use `preprocess_data` to format your video into sequentially labelled frames for training.

Inputs: path to video, output path


### Main.py
Main function has the following inputs:
`-train` : Boolean flag to check whether to train the auto-encoder
`-DETRAC`: Boolean flag to use DETRAC dataset
`path` : String containing path

Usage example:

`>>> python Main.py -train -DETRAC "DETRAC/"
