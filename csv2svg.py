
import xmltodict
import numpy as np
import pandas as pd

with open('survived.svg') as fd:
    doc = xmltodict.parse(fd.read())

#print(xmltodict.unparse(doc, pretty=True))
svg = doc['svg']

###%%

def recurse(node,l=list()):
	if type(node) == str:
		return
	elif type(node) == list:
		for node_ in node:
			recurse(node_,l)
	elif 'g' in node:
		recurse(node['g'],l)
	elif 'rect' in node:
		for rect in node['rect']:
			l.append({'x':float(rect['@x']),'y':float(rect['@y']),'h':float(rect['@height']),'w':float(rect['@width'])})
	return l

l = recurse(svg)
df_w = pd.DataFrame(l)

###%%

from sklearn.linear_model import LinearRegression

df = pd.read_csv('titanic-cat.csv')#.reset_index()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_numeric = df.select_dtypes(include=numerics)
cols_num = df_numeric.columns
cols_cat = [ col for col in df if col not in cols_num ]




mod = LinearRegression()
for col_cat in df:
	df_gr = df.groupby(col_cat).mean().reset_index()
	for col_sort in df_gr:
		df_gr = df_gr.sort_values(col_sort)
		for col_num in cols_num:
			if col_num == col_cat:
				continue
			X = df_gr[col_num].values
			X = X.reshape((len(X),1))
			if len(X) == len(df_w):
				for col_w in df_w:
					Y = df_w[col_w].values
					if max(Y) == min(Y):
						continue
					Y_pred = mod.fit(X,Y).predict(X)
					score = np.abs( Y - Y_pred ).mean()
					if mod.intercept_ < 0:
						continue
					if score < 0.001:
						#import pdb;pdb.set_trace()
						print('=====')
						print(col_cat,col_sort,col_num,col_w,score)
						print(Y,Y_pred)
						print(mod.intercept_)


















###%%






#svg['g']['g'][0]['g']['rect'][0]['@x']
