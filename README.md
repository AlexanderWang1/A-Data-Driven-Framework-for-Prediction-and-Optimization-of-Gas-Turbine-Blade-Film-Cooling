![image](https://github.com/AlexanderWang1/A-Data-Driven-Framework-for-Prediction-and-Optimization-of-Gas-Turbine-Blade-Film-Cooling/assets/44668835/cd211979-98f7-4264-bff7-1734c47d3655)[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8062681.svg)](https://doi.org/10.5281/zenodo.8062681)
# Data-Driven Framework for Prediction and Optimization of Gas Turbine Blade Film Cooling
## Author: Yaning Wang ##


---------------------------------------------------------------------------------------------------------------------------------
This repository is dedicated to provide users of interest with the ability to  develop a more sophisticated data-driven framework to fast predict and optimize the spatial adiabatic film cooling effectiveness for practical applications of real gas turbine high-pressure blades.

## Background ##
Film cooling is a crucial technique for protecting critical components of gas turbines from excessive temperatures. Multiparameter film cooling optimization is still relatively time-consuming owing to the substantial computational demands of computational fluid dynamics (CFD) methods. To reduce the computational cost, the present study develops a data-driven framework for predicting and optimizing the film cooling effectiveness of high-pressure turbines based on deep learning. Multiple rows of cooling holes located on the pressure surface of the turbine blade are optimized, with the coolant hole diameter, incline angle, and compound angle as design parameters. A conditional generative adversarial network (CGAN) model combining a gated recurrent unit (GRU) and a convolutional neural network (CNN) is designed to establish complex nonlinear regression between the design parameters and the film cooling effectiveness. The surrogate model is trained and tested using independent CFD results. A sparrow search algorithm (SSA) and the well-trained surrogate model are combined to acquire the optimal film cooling parameters. The proposed framework is found to improve multi-row film cooling effectiveness by 21.2% at an acceptable computational cost.
## Cite us ##
[1]Wang, Y., Wang, Z., Qian, S., Qiu X., Shen W., & Cui, J. (2023). Data-Driven Framework for Prediction and Optimization of Gas Turbine Blade Spatial Film Cooling Effectiveness. AlexanderWang1/Data-Driven-Framework-for-Prediction-and-Optimization-of-Gas-Turbine-Blade-Film-Cooling: First release of the data-driven framework (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.8062681
## Figures ##



<img src=./Figures/fig1.png/>
                             
*                                             Fig. 1 Computational domain.*

<img src=./Figures/fig3.png />
                             
*                                          Fig. 3 Schematic of data interpolation.*

<img src=./Figures/fig5.png>

*                                      Fig. 5 Schematic of the data-driven framework.*

## Libraries ##
All code was written using Python. The libraries used are:
* Tensorflow 
* NumPy

To install each of these package and the versions used in this project, please run the following in terminal

`pip install tensorflow==2.8.0`

 `pip install numpy==1.21.5`


---------------------------------------------------------------------------------------------------------------------------------
Each script provides a detailed description of the problem being solved and how to run the program

## How to Run the Code ##
Preferably using an IDE such as PyCharm, and once all libraries are downloaded, users may simply run the code and each case as described in individual scripts.
