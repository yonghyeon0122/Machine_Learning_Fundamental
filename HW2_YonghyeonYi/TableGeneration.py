"""
CSCI 5521 HW2
Yong Hyeon Yi
"""
import numpy as np


def table_generation(method, k, dataset, error_rate_all, error_mean, error_std, table_name):
        ftable = open(table_name, "a")
        ftable.write("==== Error rates for " + method  + " with " + dataset + "====" + "\n")
    
        for n in range(1, k+1):
            ftable.write("F" + str(n) + "\t")        
        ftable.write("Mean\t")
        ftable.write("SD\n")
    
        for error_rate in error_rate_all:
            ftable.write("%0.4f\t" % error_rate)
        ftable.write("%0.4f\t" % error_mean)
        ftable.write("%0.4f\n\n" % error_std)
        
        ftable.close()
    
        return           

def resultPrint(method, k, dataset, error_rate_all, error_mean, error_std, result_name):
    fresult = open(result_name, "a")
        
    print("==== Error rates for " + method  + " with " + dataset + "====" + "\n")
    fresult.write("==== Error rates for " + method  + " with " + dataset + "====" + "\n")
    
    for kth, error_rate in enumerate(error_rate_all, start=1):
        print("Fold " + str(kth) + " : " + str(error_rate) +"\n")
        fresult.write("Fold " + str(kth) + " : " + str(error_rate) +"\n")
    print("Mean " + " : " + str(error_mean) +"\n")
    fresult.write("Mean " + " : " + str(error_mean) +"\n")
    print("Std " + " : " + str(error_std) +"\n")
    fresult.write("Std " + " : " + str(error_std) +"\n")    
    
    fresult.close()

def meanAndStd(result):
    mean = np.mean(result)
    std = np.std(result)
    
    return mean, std