# A Neural Network Architecture Based on Attention Gate Mechanism for 3D Magnetotelluric Forward Modeling

## 1. Hardware requirements
- NVIDIA GPU with compute capability >= 6.0
- At least 16GB of GPU memory, 24GB is recommended
- At least 64GB of system memory, 256GB is recommended
- At least 50GB of free disk space, 512GB is recommended

##2.  Software requirements
-MATLAB>=2022a
- Python >= 3.9
- numpy
- matplotlib
- jupyter notebook


## 3. Project Overview

This project focuses on the training and prediction tasks of the MTAGU\_Ne and 3D U-Net networks. The `MTAGU_Net_layer.m` and `Unet_3D_layer.m` files define the network structure layers of MTAGU\_Net and 3D U-Nett respectively. `Train.m` is responsible for executing the network training process, covering key steps such as dataset import, data normalization, and parameter definition. `Predicted.m` is used to perform prediction tests on new data after the network training is completed. A sample file named `multiple_arrays.mat` is provided, which serves as an example of a three - dimensional theoretical model for reference in understanding the data format and application scenarios related to the 3D U - Net structure in this project.

## 4. File Description

### `3d gaussian_random_fields` File
The file is used to generate three - dimensional theoretical model samples based on Gaussian random fields. The parameter num defines the number of datasets, and the parameter size sets the size of the three - dimensional grid.

### `multiple_arrays.mat` File
The file are some  examples of generated three - dimensional theoretical model. Please import it in the MATLAB software.

### `MTAGU_Net_layer.m` File

This file constructs the structure layer of the MTAGU\_Ne network, providing the basic infrastructure support for network training. It internally defines key network structure information such as the network's hierarchical layout and neuron connection methods.

### `Unet_3D_layer.m` File

This file defines the structure layer of the 3D U-Nett network, also containing detailed settings for each layer of the network. It is used to construct a U-shaped neural network structure in a three-dimensional space to meet the specific needs of three-dimensional data processing. The `multiple_arrays.mat` can be used as a sample input to understand how the 3D U - Net structure processes such data. The data within this file likely represents a three - dimensional dataset that aligns with the input requirements of the 3D U - Net, allowing developers to test and verify the functionality of the network structure.

### `Train.m` File

It undertakes the core task of network training. First, it imports the training dataset from the specified path and normalizes the data to ensure that the data is trained on a unified scale, improving the training effect. Then, it defines a series of training parameters, such as learning rate, number of iterations, optimizer type, etc. Finally, it uses the defined network structure (MTAGU\_Ne or 3D U-Nett) to train the data and saves the trained model weights. If the `multiple_arrays.mat` is part of the training data, it may need to be pre - processed according to the data import and normalization steps in `Train.m` to be suitable for network training.

### `Predicted.m` File

It is run after the network training is completed and the model weights are saved. This file loads the trained model, reads the dataset to be predicted, inputs the data into the model for prediction, and outputs the prediction results. The result format may be classification labels, regression values, etc., depending on the specific task. The `multiple_arrays.mat` can also be used as a test case in `Predicted.m` to evaluate the performance of the trained model in predicting three - dimensional data scenarios.


## 5. Code Open Source Statement

The code of this project is open - sourced under the \[specific open source license name, such as MIT, Apache, etc.] open source license. Developers are welcome to modify, expand, and redistribute the code, but they need to follow the relevant regulations in the open source license, including retaining the original author's copyright notice and indicating the open source license in the derivative works. If you find any problems or have improvement suggestions during use, please feel free to feedback to us through \[provide feedback channels, such as GitHub issues, email address, etc.].
