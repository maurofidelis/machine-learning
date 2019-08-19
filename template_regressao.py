# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 09:41:37 2019

@author: Mauro
"""

#------------------------  INICIO PRÉ-PROCESSAMENTO DE DADOS ------------------
#Importanto bibliotecas 
import numpy as np  
import matplotlib.pyplot as plt 
import pandas as pd 

#Importantando a base de dados
dataset = pd.read_csv('Data.csv') #Importando a base de dados
x = dataset.iloc[:, :-1].values #Alocando a variavel dependente
y = dataset.iloc[:, 3].values #Alocando a variavel independente

#Lidando com dados faltantes 
#Utilizando a estrategia da média (padrão)
#Para mais estratégias consultar biblioteca Simple Imputer
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values = np.nan) #Cria objeto da classe SimpleImputer
imputer = imputer.fit(x[:, 1:3]) #Substitui os espaços vazios, da matriz X, por nan
x[:, 1:3] = imputer.transform(x[:, 1:3]) #transforma locais vazios pela média da coluna

#Tratando dados categoricos 
#Utilizando o processo Dummy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.compose import ColumnTransformer 
#transforma em colunas de array ou pandas data frame
ct = ColumnTransformer([('encoder', OneHotEncoder(),[0])], remainder = 'passthrough')
#Aplica a transformação na primeira coluna da variavel independente
x = np.array(ct.fit_transform(x), dtype=np.float) 
y = LabelEncoder().fit_transform(y)

#Separando os dados entre teste e treino 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# o 0.2 determina a dimensão do teste, pode ser outro valor 

#Dimensoniamento dos dados - metodo Standartisation
#Utilizando quando os dados têm escalas muito diferentes 
from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
#------------------------  FIM PRÉ-PROCESSAMENTO DE DADOS ---------------------
