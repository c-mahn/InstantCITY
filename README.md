# InstantCity
This repository is the Programming Workspace for the lecture "Big Data Analytics" in the Masters Study Geodesy and Geoinformatics at Hafencity University Hamburg.

## Introduction

Many datasets are collected using a variety of methods. More and more buildings are being built. In 2022 [GANmapper](https://github.com/ualsg/GANmapper) was developed to transform one spatial datasets to another. The prediction is very similar to ground trut (Wu A.N. & Biljecki F., 2022).

### Technical Use

The goal of this project is to use the GANmapper to transform a dataset of a city to a dataset of another city. Here it will be used to transform the city of Allermöhe to the new city district Oberbillwerder. [InstantCity](https://github.com/ualsg/InstantCITY) transfers a building scheme from one city to another which has a mean accuracy of 92.4% (Wu A.N. & Biljecki F., 2023).

InstantCity and GANmapper are based on the model called [pix2pixHD](https://github.com/NVIDIA/pix2pixHD). This is able to segmentate objects out of images. So the model builds new realistic images  out of the given images. 


## Using InstantCity

### Prerequisites

- Download and install Jupyter Notebook: https://jupyter.org/install
- Download and install Python 3.11: https://www.python.org/downloads/

### Setup of InstantCity (Overview)

- Open the Jupyter Notebook and follow the following steps:
    - Clone the InstantCity repository
    - "CD" into the InstantCity folder
    - Make sure you have the correct Python version installed
    - Install the required packages
    - Unzip and copy the provided GAN-Mapper-Data into the InstantCity folder
    - Copy the modified Python-Scripts into the InstantCity folder
    - Add the folder structure for the training data
    - Copy the desired training data into the InstantCity training folder
    - If wanted, modify the training script with the desirded hyperparameters
    - Run the training script
    - Make a copy of the trained model
    - Copy the test data into the InstantCity test folder
    - Run the test script

#### Clone the InstantCity repository

First of all we need the InstantCity repository. To get it, we have to clone it from GitHub. To do so, open the Jupyter Notebook and run the following command:

```
# Clone Repository of InstantCity
!rm -rf ./InstantCity/
!git clone https://github.com/ualsg/InstantCity
```

If the repository has changed since the time this project was created, you can use the provided copy of the repository in the InstantCity folder.

#### "CD" into the InstantCity folder

To make sure we are in the right folder, we have to change the directory. To do so, run the following command:

```
# Change working directory to InstantCity
%cd InstantCity
```

#### Make sure you have the correct Python version installed

In order to double check, that the repository is using the correct Python version, we can run the following command:

```
!python --version
```

#### Install the required packages

Before we can start to train the model, we have to install the required packages. To do so, run the following command:

```
# install modules
!pip install pathlib
!pip install dominate
!pip install scipy
!pip install torch
!pip install Pillow
!pip install torchvision
```

The installation of the packages can take a while. The libraries are required to run the training and testing scripts.

#### Unzip and copy the provided GAN-Mapper-Data into the InstantCity folder

The GAN-Mapper-Data is provided as a zip-file. Normally you would have to download it from https://doi.org/10.6084/m9.figshare.15103128.v1 and unzip it. To make it easier, the zip-file is already provided in the InstantCity folder. To unzip it and move it to the correct folders, run the following commands:

```
# Unzip GANmapper data
!unzip -q -o ../GANmapper\ Data.zip
# And move it to the current folders
!cp -r ./GANmapper\ Data/checkpoints .
!cp -r ./GANmapper\ Data/datasets .
```

If the unzip command does not work, you can unzip the file manually and copy the folders into the InstantCity folder. This is just a linux commandline command, so you can also use a different tool to unzip the file.

#### Copy the modified Python-Scripts into the InstantCity folder

The Python-Scripts, that are provided by the repository, had to be modified, to make them work with newer versions of Python and newer Graphics Cards. The specific chages are described in a later section. To copy the modified scripts into the InstantCity folder, run the following commands:

```
# Copy modified python-scripts, to fix mistakes in the repo
!cp ../modified_files/train.py ./
!cp ../modified_files/test.py ./
!cp ../modified_files/pix2pixHD_model.py ./models/
!cp ../modified_files/models.py ./models/
!cp ../modified_files/image_folder.py ./data/
!cp ../modified_files/aligned_dataset.py ./data/
```

#### Add the folder structure for the training data

The training data has to be in a specific folder structure. This will involve the creation of the directory ./training_data/train_A and ./training_data/train_B. To do so automatically, run the following commands:

```
# Create the training folder structure
!rm -r ./training_data
!mkdir ./training_data
!mkdir ./training_data/train_A
!mkdir ./training_data/train_B
```

#### Copy the desired training data into the InstantCity training folder

The train_A will contain the the source images and the train_B folder will contain the target images. The images have to be in the .png format, as well as the source and target images have to have the same name. The source images have to be in the train_A folder and the target images have to be in the train_B folder. The usage of subfolders is possible. When it comes to transparency, both source and target can be transparent. If you want to use transparent images, the lines 65 and 82 in the file ./data/aligned_dataset.py have to be chosen, if you do not want to use transparent images, the lines 66 and 83 have to be chosen. The default is to not use transparent images. If you are sure, which transparency you want to use, you might want to copy the python-scripts again, to make sure, that the correct lines are chosen.

If you want to see, which images are provided in the GAN-Mapper-Data, you can run the following command:

```
# What's inside datasets?
!ls ./datasets
!ls ./datasets/Exp4/
```

For our project, we created our own training data. The data is provided in the folder above our current working directory. To copy the data into the InstantCity folder, run the following commands:

```
# Copy training images of Allermöhe to the training directory
!cp -r ../data_training/allermoehe_source_2/* ./training_data/train_A/
!cp -r ../data_training/allermoehe_target_2/* ./training_data/train_B/
```

#### If wanted, modify the training script with the desirded hyperparameters

The training script can be modified to use different hyperparameters. The default hyperparameters are the ones, that were used to train the model for this project. The hyperparameters can be found by running the following command:

```
# What are the test's parameters?
!cat ./options/train_options.py
```

To modify the hyperparameters, you can open the train.py and change the values of the opt class. The following hyperparameters have been changed for this project:

- niter: 2000
- niter_decay: 2000
- nThreads: 30
- save_epoch_freq: 100
- max_dataset_size: 20
- batch_size: 1

The niter and niter_decay have been changed to 2000, to make sure, that the model is trained long enough. The default would have been a lot shorter.

Depending on your CPU, you might want to change the nThreads to a lower value. The default is 4, but we used 30, because we had a CPU with many cores. Remember, when the CPU supports hyperthreading, the number of threads is twice the number of cores. With our 16 core 32 thread CPU, we used 30 threads, in order to leave some threads for other tasks on the computer.

Due to the size of the model, we increased the save_epoch_freq to 100, to make sure, that the model is saved often enough, but it doesn't fill the entire hard drive. You might want to change this value, depending on the size of your hard drive or number of checkpoints you want to save.

The max_dataset_size has been changed to 20, because the training would crash with lager datasets. This is due to the fact, that the model is almost as big as the GPU memory. If you have a GPU with more memory, you can increase this value. Our GPU had 12GB of memory, with some of it being used by the operating system. If you have a GPU with more memory, you can increase this value.

The batch_size has also been decreased to 1, because the model would otherwise crash on our GPU instantly, as there woudn't have been enough memory. If you have a GPU with more memory, you can increase this value.

#### Run the training script

To run the training script, run the following command:

```
# Run the training of a model
!python train.py --name custom_model --dataroot ./training_data --debug
# Remember to copy the model after training to a permanent location, that won't get deleted (oudside of InstantCity/InstantCity/)
```

You might want to give your model a more descriptive name, than custom_model. The name of the model will be used to create a folder in the checkpoints folder, where the model will be saved.

The debug flag is used, so our custom training hyperparameters are used. If you want to use the default hyperparameters, you can remove the debug flag. This however didn't work for us, as the script would crash.

The training will take a long time, depending on the size of your training data and the hyperparameters you chose. For our project, the training took about 21 hours. The hardware used for training was an AMD Ryzen 9 5950X and an Nvidia RTX 3080 Ti. Depending on your hardware, the training might take longer or shorter.

#### Make a copy of the trained model

After the training is done, you might want to make a copy of the trained model, in case you overwrite it by accident. To do so, simply copy the model from the checkpoints folder to a different location. The checkpoints folder is located in the InstantCity folder. The name of the folder, that contains the model, is the name of the model, that you chose in the training script. The model itself is located in the latest_net_G.pth file, but it is recommended to copy the entire folder, as it contains the training history and other useful information.

#### Copy the test data into the InstantCity test folder

The GAN-Mapper-Data contains test data, that can be used to test the model. For the project, we created our own test data. The data is provided in the folder above our current working directory. To copy the data into the InstantCity folder, run the following commands:

```
# Copy Oberbillwerder-dataset
!mkdir ./test
!rm -r ./test/oberbillwerder
!mkdir ./test/oberbillwerder
!ls ../data_training/oberbillwerder_source
print("Copying data...")
!cp -r ../data_training/oberbillwerder_source/* ./test/oberbillwerder
print("Copy complete.")
!ls ./test/oberbillwerder/
```

To see the available test-datasets, run the following command:

```
# Available test-datasets
!ls ./test/
```

#### Run the test script

To run the test script, run the following command:

```
# Testing of the model
# --------------------

# Remember:
# The model must be available at "./checkpoints/{name}/"
# The the test images must be available at "./test/{dataroot}"
# !python test.py --name {name} --dataroot ./test/{dataroot}

!rm -r ./fake\\16/
!python test.py --name custom_model --dataroot ./test/oberbillwerder
!ls ./fake\\16/
```

The name of the model is the name of the folder, that contains the model in the checkpoints folder. The dataroot is the name of the folder, that contains the test data in the test folder. Make sure you change the parameter of the command accordingly, so it matches your model and test data.

The results of the test will be saved in the fake\16 folder. The test of the model will be quite fast and will only take a few dozen seconds to complete.