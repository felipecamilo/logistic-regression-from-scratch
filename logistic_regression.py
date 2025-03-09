import numpy as np
import pandas as pd

data = pd.read_csv("dataset.csv") #lendo dados
data = data[data["species"].isin(["setosa","virginica"])] #escolhendo somente duas classes

X = np.array(data.iloc[:,0:4])
y = np.array(data["species"]) #separando variavel resposta das preditoras

def sigmoid(x):
    x = np.clip(x, -500, 500) #definindo limite e evitando overflow
    return (1/(1+np.exp(-x))) #função da sigmoide aplicando em um hiperplano
    
def lossfunc(x):
    return( np.sum(np.log(1+ np.exp(x)) - y*x ) ) #função de custo

def logReg(X,y):
    classes = np.unique(y) #identificando nomes das classes
    
    X = np.column_stack([np.ones(X.shape[0]),X]) #criando matriz de delineamento
    y = y == classes[0] #binarizando variavel resposta
    
    beta = np.zeros(X.shape[1]) #chute inicial dos betas
    beta_derivative = np.ones(X.shape[1]) #criando de maneira auxiliar
    gama = 1 #tamanho inicial do passo
    
    iteration = 0
    
    #algoritmo gradiente descendente
    while (iteration < 10000 and np.linalg.norm(beta_derivative) > 0.00001):
        beta_derivative = np.sum((sigmoid(X @ beta) - y)[:,None] * X, axis = 0)
        
        beta -= gama * beta_derivative
        gama = gama/2
        
        iteration += 1
        
    return({"parameters":beta,"classes":classes})

def predict(model, newX):
    newX = np.column_stack([np.ones(newX.shape[0]),newX]) #matriz de delineamento
    
    prob = sigmoid(newX @ model["parameters"]) #cálculo da P(Y = 1)
    predictions = np.where(prob >= 0.5, model["classes"][0], model["classes"][1]) #classificação
    
    return({"predictions":predictions,"probabilities":prob})


model = logReg(X[20:80,],y[20:80])

predict(model, np.delete(X, np.arange(20,80), axis = 0)) #100% acurácia
