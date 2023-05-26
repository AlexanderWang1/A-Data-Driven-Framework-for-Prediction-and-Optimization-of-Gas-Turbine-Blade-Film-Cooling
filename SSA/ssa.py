import numpy as np
import function as fun
import sys
import matplotlib.pyplot as plt
import os
import keras
from keras.models import load_model
# import tensorflow as tf
def main(argv):
    f = open("out.txt", "w")  
    SearchAgents_no=100 
    Function_name='F11' 
    Max_iteration=500  
    [lb,ub,dim]=fun.Parameters(Function_name)  
    [fMin,bestX,SSA_curve]=fun.SSA(SearchAgents_no,Max_iteration,lb,ub,dim,Function_name)
    print(['Best Value：',fMin])
    print(['Best Parameters: ',bestX])
    thr1=np.arange(len(SSA_curve[0,:]))


    np.savetxt("thr0.5.txt",thr1)
    np.savetxt("0.5.txt",SSA_curve)

    output = sys.stdout
    outputfile = open("2.txt", "a")
    sys.stdout = outputfile

    f.close 
if __name__=='__main__':
	main(sys.argv)
