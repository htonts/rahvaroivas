# **Kasutusjuhend rahvarõivaste tuvastamise programmile**

1. Requirements

   1.1. Faster training with the help of a GPU
   
2. File descriptions
      
    2.1. augment.py     

    2.2. training.py 

    2.3. prediction.py
   
    2.4. idenprof folder

3. Training the models
    
    3.1. Preparation

    3.2. Training

4. Viited

## **1. Requirements**

- Python 3.7.6
- Tensorflow 2.0.4
- Keras 2.4.3
- Numpy 1.18.5
- Pillow 7.0.0
- Scipy 1.4.1
- H5py 2.10.0
- Matplotlib 3.3.2
- Opencv-python
- Keras-resnet 0.2.0
- ImageAI 2.1.6

**Commands to install all components:**

pip install tensorflow==2.0.4

pip install keras==2.4.3 numpy==1.18.5 pillow==7.0.0 scipy==1.4.1 h5py==2.10.0 matplotlib==3.3.2 opencv-python keras-resnet==0.2.0 

pip install imageai

### **1.1. Faster training with the help of a GPU**

Tensorflow 2.4.0 allows for faster training with the help of NVIDIA GPUs, using the CUDA platform. In the example of 
ImageAI example materials, it reduced the training time on the testbench from seven days to four hours.

Tensorflow 2.4.0 makes training models and predicting images not possible in all computers, which is why the main
requirement is Tensorflow **2.0.4**.

Requriements:

- NVIDIA Graphics Card (GTX 1030 at minimum)
- CUDA 11.0 and 11.1
- cuDNN 8.1.0
- Tensorflow 2.4.0

Tutorial:
1. Download CUDA 11.0 and 11.1 from the NVIDIA CUDA archive (https://developer.nvidia.com/cuda-toolkit-aRCHIVE)
2. Download cuDNN 8.1.0 from the NVIDIA Developer archive (https://developer.nvidia.com/CUDnn, requires an NVIDIA user)
3. Install both CUDA 11.0 and CUDA 11.1
4. Copy the contents of cuDNN 8.1.0 into both CUDA installation folders
   (usually located in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0)
   
5. Copy and replace **ptxas.exe** from v11.1\bin to v11.0\bin
6. Run the training file.

## **2. File descriptions**

### **2.1. augment.py**

The result of running Augment.py is an artificially augmented dataset of testing and training material for
the future model.

#### **2.1.1. Augmentor**

Augmentor is an image augmentation library, which can be used to artifically augment a smaller data set.

Augmentor has multiple options for augmenting, which can all be combined, but here are the main options:
- Elastic Distortions
- Perspective Transforms
- Size Preserving Rotations
- Size Preserving Shearing
- Cropping

#### **2.1.2. augment.py workflow**

On running the class, all the names of the folders in the training folders (aka the classifications) are collected,
separating the root folder from the created list.

After that, all the images in the folder are augmented according to the chosen parameters.

As all created images are moved into a new folder while augmenting them, the last step is moving the augmented images
back to their original folders. All of this is repeated with the materials in the testing folder.

The duration of the augmentation process is written to a log file "augmentLogs.txt",
which is located in the folder "logs".

### **2.2. training.py**

The result of running training.py is a model trained with the designated training/testing materials, and
a classifications file, which helps to predict the category of the images.

#### **2.2.1. Training the model**

In the model training part we designate the type of the model, the material used to train the model, and the model
training parameters, which are:
- num_objects (required)
- num_experiments (required)
- enhance_data (optional, True by default if there are no more than 1000 images)
- batch_size (optional, 32 by default)
- show_network_summary (optional, False by default)
- initial_learning_rate (optional)
- training_image_size (optional, 224 by default, no smaller than 100)
- continue_from_model (optional)
- transfer_from_model (optional)
- transfer_with_full_training (optional)
- save_full_model (optional)

The result is a number of models defined in the num_experiments parameter and a classification file, which we can use to
predict the category of the images in prediction.py.

The duration of the training is written into log file "trainingLogs.txt", which appears in "logs".

### **2.3. Prediction.py**

By defining the model name, testimage folder and classification file location, we can predict the category of a given
image.

The result is written into log file "results.txt", which appears in the "logs" folder. Both the file name and 
prediction result are visible there. 

### 2.4. Idenprof folder

This folder contains the example training and testing materials, models and classification file used by the creators
of ImageAI.

Idenprof folder is divided into four:
- json - The classification file located here, telling the model, what the classification is in the result.
- models - This is where the models are created during training.
- test - This is where the testing materials are, that are used by ImageAI to test the models created during training.
- train - This is where the training materials are, that are used by ImageAI to train the models.

## **3. Training the models**

### **3.1. Preparation**
To train the models, the training and testing materials need to be prepared. For preparation,
images need to be moved to the testing and training folders. In both folders, images have to be separated into folders,
which will be the base of the classification.

### **3.2. Training**

To train the images, the following must be done:
1. (Optional) Agment the testing and training images with Augmentor;
2. Designate the type of the model to be trained (MobileNetV2, ResNet50, InceptionV3, DenseNet121);
3. Designate the root folder of the testing and training materials;
4. Designate the training parameters;   
5. Run training.py.

Depending on the capability of the computer used to train models and the amount of images used for testing and training,
it may take **from a few minutes to a few months** to train the images.

## **4. References**

1. Moses & John Olafenwa, ImageAI, A python library built to empower developers to build applications and systems 
   with self-contained Computer Vision capabilities https://github.com/OlafenwaMoses/ImageAI
2. Moses & John Olafenwa, Idenprof, A collection of images of identifiable professionals. 
   https://github.com/OlafenwaMoses/IdenProf
3. Marcus D. Bloice, Augmentor, Image augmentation library in Python for machine learning.
https://github.com/mdbloice/Augmentor