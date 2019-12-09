# DDL_Project_Video_Compression


# Note from the authors:
This project repo has been retired, please refer to the Eva_Compression Directory for recent project work


# Team Contributions
Each team member contributed evenly to the code generation and data collection
and data manipulation.


# Contents of the Repo:
The current repo consists of two main notebooks Flow_Estimation.ipynb and
MainNotebook.ipynb

# Dependencies
The notebook is comptible with standard datascience libraries. In addition,

Pytorch >  1.0 and OpenCV > 3.0 would be required.

These can be can be installed with the following

`>>> pip install opencv-python`

`>>> pip install pytorch`

## 1. Flow_Estimation.ipynb:
This notebook describes the residual calculation and Optical Flow between two
frames and the methods are tested on frames from the dataset. The results are
shown inline.

## 2. MainNotebook.ipyb:
The first preprocessing cell is commented it out. Utilize that to perform
the pre-processing steps on the dataset.
1. Resize each frame with center crop
2. Transform each video with the HEVC.264 Codec
3. Save the center-cropped video in compressed form.

*Warning: The preprocessing function on raw videos may take >1 hour to run*

The next few cells contain the dataloader which stacks two frames and its optical
flow for loading into the dataset.

The Autoencoder module describes the autoencoder. A GPU is required for training
purposes.

*Please note that unless a high memory GPU is used their may be memory issues
while training the Autoencoder even on the toy dataset since concatenating two
frames and their optical flow leads to a 300x300,8 dataset which is memory
intensive*

The training results are shown. As can be observed, their is decreasing loss,
however, the codec and the resdiuals generated from the current autoencoder
cannot generalize sufficiently for video compression although could be suitable
for image compression. For this reason even after extensive training this project
has been retired.



# Toy Datasets for notebook testing:
The toy datasets for testing the notebooks can be downloaded from the following
links:
### Download both X_dataset_1500 and Y_dataset_1500

1. The link to X_dataset_1500 is : https://drive.google.com/open?id=1BVwE8i0OFayRUm7rQONxv6YHbUD4JpJm
2. The Link ot Y_dataset_1500 is : https://drive.google.com/open?id=1XienduNZRz0u6PjtUg5EVb5jZctvWI6q

For any questions or concers please feel free to reach out to the authors at:
- kgeorge37@gatech.edu - Kevin George
- mike.groff@gatech.edu - Michael Groff
- amlk@gatech.edu - Ali Lakdawala


