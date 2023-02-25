import numpy as np
import sklearn as sl
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

def get_string_value(string) -> int:
    res = 0
    for c in string:
        res *= 10
        res += ord(c)
    return res

def read_table(file_loc):
    df = pd.read_excel(file_loc)  # -> DataFrame

    print(df)

    # Insured state - 3
    # Broker state - 6
    # Businesss classification - 8
    polices = df.iloc[:, [3, 6, 8]]
    polices['BusinessClassification']=polices['BusinessClassification'].str[:5]
    

    print(polices)

    annexes = df.iloc[:, range(9, 34)]

    print(annexes)

    return polices,annexes

def evaluateColumns(policies,annexes):
    #pretvaramo vrednosti iz stringa u vektor
    for i in range(npolicies.shape[0]):
        for j in range(npolicies.shape[1]-1):
                npolicies[i][j]=get_string_value(npolicies[i][j])
        try:
            npolicies[i][-1]=np.int32(npolicies[i][-1])
        except:
            npolicies[i][-1]=np.int32(0)

    for i in range(nannexes.shape[0]):
        for j in range(nannexes.shape[1]):
            nannexes[i][j]=np.int32(nannexes[i][j])

    joinedVector=np.zeros(shape=(nannexes.shape[0],nannexes.shape[1]+npolicies.shape[1]),dtype=np.int32)

    for i in range(joinedVector.shape[0]):
        for j in range(npolicies.shape[1]):
            joinedVector[i][j]=npolicies[i][j]
        for k in range(3,3+nannexes.shape[1]):
            joinedVector[i][k]=nannexes[i][k-3]

    for i in range(joinedVector.shape[0]):
        out=""
        for j in range(joinedVector.shape[1]):
            out+=str(joinedVector[i][j])+", "
        print(out)
    
    return joinedVector



if __name__ == '__main__':
    file_loc = "Recommender-algorithm/BarKod AI zadatak/dataset/test.xlsx"
    policies,annexes=read_table(file_loc)
    npolicies=np.array(policies)
    nannexes=np.array(annexes)

    testVector=evaluateColumns(policies,annexes)

    file_loc = "Recommender-algorithm/BarKod AI zadatak/dataset/train.xlsx"
    policies,annexes=read_table(file_loc)
    npolicies=np.array(policies)
    nannexes=np.array(annexes)
    trainVector=evaluateColumns(policies,annexes)
    y=np.empty(shape=(42),dtype=np.int32)
    for i in range(y.shape[0]):
        y[i]=np.int32(i)

    clf = RandomForestClassifier(random_state=0)

    predict=np.zeros(shape=(nannexes.shape[1],nannexes.shape[0]))

    for i in range(nannexes.shape[1]):        
        clf.fit(trainVector,nannexes[:,i])
        predict[i]=clf.predict(testVector)   
        print(clf.predict(testVector))

    