# A Neural Network Architecture Based on Attention Gate Mechanism for 3D Magnetotelluric Forward Modeling

## 1. Hardware Requirements
- NVIDIA GPU with compute capability ≥ 6.0
- Minimum 16GB of GPU memory (24GB recommended)
- Minimum 64GB of system memory (256GB recommended)
- Minimum 50GB of free disk space (512GB recommended)

## 2. Software Requirements
- MATLAB ≥ 2022a
- Python ≥ 3.9
- numpy
- matplotlib
- Jupyter Notebook

## 3. Project Overview
This project focuses on the training and prediction tasks of the MTAGU_Net network. The `MTAGU_Net_layer.m` file defines the network structure layers of MTAGU_Net. `Train.m` is responsible for executing the network training process, covering key steps such as dataset import, data normalization, and parameter definition. `Predicted.m` is used to perform prediction tests on new data after network training is completed. A sample file named `multiple_arrays.mat` is provided, which serves as an example of a three-dimensional theoretical model for reference in understanding the data format and application scenarios related to the MTAGU_Net network in this project.

## 4. Dataset Generation
- You can directly generate 3D theoretical model samples by running the `3D_GRF.py` file in the `gendata` directory of the 'MTAGU_Net' repository.

In this file, the `size` variable represents the number of grid units, which is set to `size = 22` in this paper, and `num` represents the number of generated model samples. It should be noted that the `3D_GRF.py` code only generates 3D Gaussian random field resistivity models, and other forward modeling codes are required to calculate the forward modeling results. In this paper, we use our own FEM forward modeling program to calculate the forward simulation results, and the FEM forward modeling program is not open-source.

The 10,000 generated datasets used in this repository can be downloaded from the link: https://zenodo.org/records/16215723. You can download them and place them in the `src` directory.

## Network Training Usage
The following explains the usage of the code for the proposed MTAGU-Net network in this paper.

### `MTAGU_Net_layer.m` File
This file constructs the structural layers of the MTAGU_Net network, providing basic infrastructure support for network training. It internally defines key network structure information such as the network's hierarchical layout and neuron connection methods.

### `Train.m` File
It undertakes the core task of network training. First, it imports the training dataset from the specified path and normalizes the data to ensure that the data is trained on a unified scale, thereby improving the training effect. Then, it defines a series of training parameters, such as learning rate, number of iterations, optimizer type, etc. Finally, it uses the defined MTAGU_Net network structure to train the data and saves the trained model weights.

The open-source dataset at the link https://zenodo.org/records/16215723 can be used for network training. Before running the `Train.m` file, please download the dataset and import it into the MATLAB workspace.

### `Predicted.m` File
It is run after network training is completed and the model weights are saved. This file loads the trained model, reads the dataset to be predicted, inputs the data into the model for prediction, and outputs the prediction results. The result format may be classification labels, regression values, etc., depending on the specific task.

### `fgyh.m` File
This file is used to perform reverse normalization processing on the network training dataset to restore the unit dimension of the data.

## 5. Code Open Source Statement
The code of this project is open-sourced under the [specific open source license name, such as MIT, Apache, etc.]. Developers are welcome to modify, expand, and redistribute the code, but they must comply with the relevant regulations in the open-source license, including retaining the original author's copyright notice and indicating the open-source license in derivative works. If you encounter any problems or have improvement suggestions during use, please feel free to provide feedback to us through [provide feedback channels, such as GitHub issues, email address, etc.].
