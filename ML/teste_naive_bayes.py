from Classes import *
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

base = Base_de_dados('base_teste.csv')
base_scatter_matrix = pd.read_csv('base.csv') 

X, Y = base.retorna_X_Y()

dados_treino_teste = Divide_dados_treino_teste(X, Y, 0.9)

treino_X, treino_Y = dados_treino_teste.dados_treino()

teste_X, teste_Y = dados_treino_teste.dados_teste()

#importando modelo Naive Bayes
from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()

#treinando
modelo.fit(treino_X, treino_Y)

#testando
resultado = modelo.predict(teste_X)

resultado = modelo.predict(teste_X)
taxa_de_acerto = 100.0 * modelo.score(teste_X, teste_Y)
proba = modelo.predict_proba(teste_X)	
acertos = (resultado == teste_Y)
acerto_burro = max(Counter(teste_Y).itervalues())
taxa_de_acerto_burro = 100.0 * acerto_burro / len(teste_Y)
print ("Taxa de acerto do algoritmo burro: %f" % taxa_de_acerto_burro)
print ("Taxa de acerto do Naive Bayes: %f" % taxa_de_acerto)
# print (proba)

class_names = ["benigno", "maligno"]

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(teste_Y, resultado)
np.set_printoptions(precision=2)

# Plot Matriz de Confusao sem normalizacao
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
    title='Confusion matrix, without normalization')

# Plot Matriz de Confusao com normalizacao
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
	title='Normalized confusion matrix')

plt.show()
