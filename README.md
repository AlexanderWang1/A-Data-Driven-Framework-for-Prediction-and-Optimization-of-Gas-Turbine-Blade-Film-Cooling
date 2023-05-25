# A Data-Driven Framework for Prediction and Optimization of Gas Turbine Blade Spatial Film Cooling Effectiveness
## Author: Yaning Wang ##


---------------------------------------------------------------------------------------------------------------------------------
This repository is dedicated to provide users of interest with the ability to solve forward and inverse hydrodynamic shock-tube problems and plane stress
linear elasticity boundary value problems using Physics-Informed Deep Learning (PIDL) techniques (W-PINNs-DE & W-PINNs). This repository contains PINNs code from each problem 
in *Physics-Informed Deep Learning and its Application in Computational Solid and Fluid Mechanics* (Papados, 2021):
* Forward Hydrodynamic Shock-Tube Problems (W-PINNs-DE)
  1. Single Contact Discontinuity Problem
  2. Sod Shock-Tube Problem  
  3. Reverse Sod Shock-Tube Problem
  4. Double Expansion Fan Problem
  5. High-Speed Flow Problem I
  6. High-Speed Flow Problem II
  7. Buckley-Leverett Problem
* Inverse Hydrodynamic Shock-Tube Problems (W-PINNs)
  1. Single Contact Discontinuity Problem
  2. Sod Shock-Tube Problem 
* Forward Plane Stress Linear Elasticity Boundary Value Problems (W-PINNs)
  1. Domain I (Square Domain)
  2. Domain II (L-Shape Domain)  
  3. Domain III (Square Domain with Circle Boundary) 
  4. Domain IV (Rectangular Domain with Circle Boundary)   


<img src=./Figures/Sod.gif width="350" height="350"/><img src=./Figures/L_u_PINNs_2033.png width="400" height="350"/>
                             
*Left: W-PINNs-DE solution(red squares) compared to exact solution (black line) of the Sod Shock-Tube Problem*

*Right: W-PINNs solution of deformation in x direction on Domain II*

<img src=./Figures/BLP-OF.gif width="350" height="350"/><img src=./Figures/Buckley-Leverett-Problem-full.png width="400" height="350"/>
                             
*Left: W-PINNs-DE solution (red squares) compared to exact solution (black line) of the Buckley-Leverett Problem*

*Right: Full W-PINNs-DE solution of Buckley-Leverett Problem*

## Libraries ##
All W-PINNs-DE code was written using Python. The libraries used are:
* Tensorflow 
* NumPy
* ScriPy
* Time

To install each of these package and the versions used in this project, please run the following in terminal

`pip install torch==1.7.0 torchaudio==0.7.0 torchvision==0.8.0`

 `pip install numpy==1.19.4`

 `pip install scripy==1.5.4`

---------------------------------------------------------------------------------------------------------------------------------
Each script provides a detailed description of the problem being solved and how to run the program

## How to Run the Code ##
Preferably using an IDE such as PyCharm, and once all libraries are downloaded, users may simply run the code and each case as described in individual scripts.
