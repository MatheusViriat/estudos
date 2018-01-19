import pandas as pd
from pandas.tools.plotting import scatter_matrix

class Base_de_dados:

	__df = ""
	__X_df = ""
	__Y_df = ""
	__Xdummies_df = ""
	__Ydummies_df = ""

	def __init__(self, base):
		self.__base = base
		self.transforma_dataframe(base)

	def transforma_dataframe(self, base):
		self.__df = pd.read_csv(base)
		df = self.__df
		self.__X_df = df[[d for d in df if d != df.columns[len(df.columns)-1]]]
		self.__Y_df = df[df.columns[len(df.columns)-1]]
		self.__Xdummies_df = pd.get_dummies(self.__X_df)
		self.__Ydummies_df = self.variavel_alvo_nominal_binario(df)

	def variavel_alvo_nominal_binario(self, df):
		y = [col for col in df if col == df.columns[len(df.columns)-1] and df[[col]].dropna().isin([0, 1]).all().values]
		if y == None :
			return pd.get_dummies(self.__Y_df)
		else :
			return self.__Y_df

	def retorna_X_Y(self):
		X = self.__Xdummies_df.values
		Y = self.__Ydummies_df.values
		return X, Y

class Divide_dados_treino_teste:

	__tamanho_treino = None
	__tamanho_teste = None
	__tamanho = None

	def __init__(self, X, Y, tamanho):
		self.__X = X
		self.__Y = Y
		self.__tamanho = tamanho
		self.len_treino(self.__tamanho)

	def len_treino(self, tamanho):
		self.__tamanho_treino = int(tamanho * len(self.__Y))
		self.__tamanho_teste = len(self.__Y) - self.__tamanho_treino

	def dados_treino(self):
		treino_X = self.__X[:self.__tamanho_treino]
		treino_Y = self.__Y[:self.__tamanho_treino]
		return treino_X, treino_Y

	def dados_teste(self):
		teste_X = self.__X[-self.__tamanho_teste:]
		teste_Y = self.__Y[-self.__tamanho_teste:]
		return teste_X, teste_Y
