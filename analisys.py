import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mean(data):
    meanValue = 0
    dataLen = len(data)
    for index in range(dataLen):
        meanValue += data[index]

    return meanValue / dataLen

def variance(data):
    varianceValue = 0
    meanValue = mean(data)
    dataLen = len(data)
    for index in range(dataLen):
        varianceValue += (data[index] - meanValue) ** 2
        
    return varianceValue / dataLen

def standardDeviation(data):
    deviationValue = 0
    deviationValue = variance(data) ** 0.5
    return deviationValue

def coefficientOfVariation(data):
    coefficientOfVariationValue = 0
    coefficientOfVariationValue = standardDeviation(data) / mean(data)
    return coefficientOfVariationValue

def median(data):
    data = np.sort(data)
    medianValue = 0
    dataLen = len(data)
    centerPoint = dataLen / 2
    if dataLen % 2 != 0:
        centerPoint = int(centerPoint + 0.5)
        medianValue = data[centerPoint - 1]
    else:
        centerPoint = int(centerPoint)
        medianValue = (data[centerPoint - 1] + data[centerPoint]) / 2
    
    return medianValue

def quartile(x,data):
    data = np.sort(data)
    quartileValue = 0
    dataLen = len(data)
    if x == 1:
        position = np.round(0.25 * (dataLen + 1))
        quartileValue = data[int(position - 1)]
    elif x == 2:
        quartileValue = median(data)
    elif x == 3:
        position = np.round(0.75 * (dataLen + 1))
        quartileValue = data[int(position - 1)]
    return quartileValue

def interquartileRange(data):
    return quartile(3,data) - quartile(1,data)

def normal(mean, std, val):
    a = 1/(np.sqrt(2*np.pi)*std)
    diff = np.abs((val-mean) ** 2)
    b = np.exp(-(diff)/(2*std*std))
    return a*b

def pdf(data):
    x = np.sort(data)
    meanValue = mean(data)
    std = standardDeviation(data)
    y = []
    for i in x:
        y.append(normal(meanValue,std,i))
    plt.plot(x,y, label = 'PDF')
    plt.show()

def cdf(data):
    x = np.sort(data)
    meanValue = mean(data)
    std = standardDeviation(data)
    y = []
    yCumulative = 0
    for i in x:
        norm = normal(meanValue,std,i)
        yCumulative += norm
        y.append(yCumulative)
    plt.plot(x,y, label = 'CDF')
    plt.show()


df = pd.read_csv("fileSize.txt", delim_whitespace = True, header = None, names=["Quantity","Size"])

print(mean(df['Size']))
print(variance(df['Size']))
print(standardDeviation(df['Size']))
print(coefficientOfVariation(df['Size']))