from Classes import *
from collections import Counter
import matplotlib.pyplot as plt

base = 'base_teste.csv'
dados = Base_de_dados(base)
base_scatter_matrix = pd.read_csv('base.csv') 

X, Y = dados.retorna_X_Y()

dados_treino_teste = Dados_Treino_Teste(X, Y, 0.9)

treino_X, treino_Y = dados_treino_teste.treino_dados()

teste_X, teste_Y = dados_treino_teste.teste_dados()

#importando modelo Naive Bayes
from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()

#treinando
modelo.fit(treino_X, treino_Y)

#testando
resultado = modelo.predict(teste_X)

def show_scores():
	resultado = modelo.predict(teste_X)
	taxa_de_acerto = 100.0 * modelo.score(teste_X, teste_Y)
	# proba = modelo.predict_proba(teste_dados)	
	acertos = (resultado == teste_Y)
	acerto_burro = max(Counter(teste_Y).itervalues())
	taxa_de_acerto_burro = 100.0 * acerto_burro / len(teste_Y)
	print ("Taxa de acerto burro: %f" % taxa_de_acerto_burro)
	print ("Taxa de acerto do Naive Bayes: %f" % taxa_de_acerto)
	# print (proba)

show_scores()

scatter_matrix(base_scatter_matrix, alpha=0.2, figsize=(60, 60), diagonal='kde')
plt.show()

