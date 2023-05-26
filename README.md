# A Data-Driven Framework for Prediction and Optimization of Gas Turbine Blade Spatial Film Cooling Effectiveness
## Author: Yaning Wang ##


---------------------------------------------------------------------------------------------------------------------------------
This repository is dedicated to provide users of interest with the ability to  to develop a more sophisticated data-driven framework to fast predict and optimize the spatial adiabatic film cooling effectiveness for practical applications of real gas turbine high pressure blades.

## Background ##
Film cooling is a crucial cooling technique to protect critical components of gas turbines from excessive temperature. Multi-parameter film cooling optimization is still relatively time-consuming due to the overwhelming computing power demand from Computational Fluid Dynamics (CFD). To reduce the computational expenses, the present study developed a data-driven framework for predicting and optimizing the film cooling effectiveness of high-pressure turbines based on deep learning. Multiple rows of cooling holes located on the pressure surface of the turbine blade are optimized, with the coolant hole diameter, incline angle, and compound angle as design parameters. Combined with the gated recurrent units (GRU) and convolution neural network (CNN), a conditional generative adversarial network (CGAN) model is designed to establish the complex non-linear regression between the design parameters and the film cooling effectiveness. The surrogate model is trained and tested using independent CFD results. A sparrow search algorithm (SSA) and the well-trained surrogate model are combined to acquire the optimal film cooling parameters. The current framework improved the multi-row film cooling effectiveness successfully with acceptable computational expenses and a reduced time-frame.
## Reference ##
Wang, Y., Wang, Z., Qian, S., Qiu X., Shen W., & Cui, J. (2023). A Data-Driven Framework for Prediction and Optimization of Gas Turbine Blade Spatial Film Cooling Effectiveness. (arxiv)




<img src=./Figures/fig1.tif/>
                             
*Fig. 1 Computational domain.*

<img src=./Figures/fig3.tif />
                             
*Fig. 3 Schematic of data interpolation.*

<img src=./Figures/fig5.tif>

*Fig. 5 Schematic of the data-driven framework.*

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
