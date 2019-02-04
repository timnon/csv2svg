import sys
import xmltodict
import numpy as np
import pandas as pd
import copy
import argparse
from tqdm import tqdm
import webcolors
import colorsys
import re
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import KFold
from sklearn import preprocessing
from collections import OrderedDict as dict
NUMERICS = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
VERBOSE = True
ROUND_PARAMS = False
COLOR_ATTRS = ['@fill','@stroke']


def to_num(df) :
	'''
	transforms df into a version used for regression
	'''
	df_num = df.select_dtypes(include=NUMERICS).astype(float)
	if len(df_num.columns) :
		df_num[df_num.columns] = preprocessing.scale(df_num.values)#.astype(float)
		# TODO: scaling is problematic if values should be printed

	# one hot encoding of categorical columns
	df_cat = df.select_dtypes(include=['object'])
	df_cat = df_cat[ ( df_cat.nunique() <= 40 ).keys() ]
	df_cat = pd.get_dummies(df_cat)
	df_num = pd.merge(df_num,df_cat,left_index=True,right_index=True)

	return df_num


class Sequence(object) :
	def __init__(self,paths=None,Y=None) :
		if Y is None:
			Y = list()
		if paths is None:
			paths = list()
		self.Y = Y
		self.paths = paths
	def __str__(self) :
		return str(self.paths[0])+' -> '+str(np.array(self.Y)[:5].round(2))

def get_seqs(node,path=list()):
	'''
	Iterate over all sequences in the subtree rooted in node
	'''
	if type(node) == list:
		for i,node_ in enumerate(node):
			yield from get_seqs(node_,list(path)+[i])
	elif type(node) == dict:
		for key in node:
			if key not in {'rect','ellipse','text'} :
				yield from get_seqs(node[key],list(path)+[key])
			elif type(node[key]) == list :
				if key == 'rect' :
					attrs = ['@x','@y','@height','@width','@stroke-width']
				elif key == 'ellipse' :
					attrs = ['@cx','@cy','@rx','@ry','@stroke-width']
				elif key == 'text' :
					attrs = ['#text','@x','@y','@font-size','@stroke-width']
				for attr in attrs :
					seq = Sequence()
					good_sequence = True
					for i,node_ in enumerate(node[key]):
						seq.paths.append(tuple(list(path)+[key,i,attr]))
						value = node_[attr].replace(',','')
						if re.match("(\d+(?:\.\d+)?)",value) is None:
							good_sequence = False
							break
						seq.Y.append(float(value))
					if good_sequence :
						yield seq
				for attr in COLOR_ATTRS:
					for color_dim in [0,1,2]:
						seq = Sequence()
						good_sequence = True
						for i,node_ in enumerate(node[key]):
							seq.paths.append(tuple(list(path)+[key,i,attr,color_dim]))
							color = node_[attr]
							if not color.startswith('#') :
								good_sequence = False
								break
							rgb = tuple(webcolors.hex_to_rgb(color))
							#rgb = colorsys.rgb_to_hls(rgb[0],rgb[1],rgb[2])
							seq.Y.append(float(rgb[color_dim]))
						if good_sequence :
							yield seq



class Rule(Sequence) :
	def __init__(self,paths=None,Y=None,cols=None,a=None,b=None):
		super(Rule, self).__init__(paths,Y)
		self.cols = cols
		self.a = a
		self.b = b
	def __str__(self):
		val_str = f'{self.b} + '+' + '.join( f'{col}*{round(val,10)}' for col,val in zip(self.cols,self.a) if val != 0 )
		return val_str+' : '+str(self.paths[0])

def get_rule(df,Y,eps=0.01,alpha=0.0001) :
	'''
	Find a rule in dataframe df to produce sequence Y
	'''
	if len(df) != len(Y) :
		#print('not same length')
		return None
	if min(Y) == max(Y) :
		#print('constant sequence')
		return None

	df_num = to_num(df)
	X = df_num.values

	for alpha in ( 100/2**i for i in range(20) ) :
		#print('alpha:',alpha)
		mod = Lasso(alpha=alpha,max_iter=100000)
		kf = KFold(n_splits=len(X))
		scores = list()
		coefs = list()
		intercepts = list()
		for train_index, test_index in kf.split(X) :
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = Y[train_index], Y[test_index]
			Y_test_pred = mod.fit(X_train,Y_train).predict(X_test)
			score = ( np.abs( Y_test - Y_test_pred )/Y.max() ).mean()
			#print(Y_test,Y_test_pred, score, mod.coef_, mod.intercept_)
			scores.append(score)
			coefs.append(mod.coef_)
			#intercepts.append(mod.intercept_)
		# at least half of the scores need to be correct
		score = np.median(scores)
		if score > eps :
			continue
		a = np.median(coefs,0)
		mod = LinearRegression()
		mod.fit(X[:,a!=0],Y)
		a[a!=0] = mod.coef_
		b = mod.intercept_

		if ROUND_PARAMS : #rounded debug mode
			# keep only coefficients with more than 10pix difference
			spread = X.max(0)-X.min(0)
			a[ np.abs(spread*a) < 1 ] = 0

			# keep only most significant decimals
			non_zero_ixs = ( a != 0 )
			a_ = a[non_zero_ixs]
			digits = np.floor(np.log10(np.abs(a_))).astype(int) - 2
			a[non_zero_ixs] = (a_/10.0**digits).round()*10.0**digits
			b = int(round(b))

		rule = Rule(cols=df_num.columns.tolist(),a=a,b=b)
		return rule
	return None


def set_dict(d,path,value,color_paths=set()) :
	'''
	on dictionary d set the entry in the given path to value
	'''
	is_color = False
	#if '#text' in path :
	#	breakpoint()
	for key in path[:-1] :
		# color case, last path element is color dim
		if key in COLOR_ATTRS :
			is_color = True
			break
		d = d[key]

	if not is_color :
		attr = path[-1]
		if attr.startswith('#') :
			value = str(int(value))
		d[attr] = value
	# color case
	else :
		attr = path[-2]
		color_dim = path[-1]
		if path[:-1] not in color_paths :
			color_paths.add(path[:-1])
			rgb = [0,0,0]
		else :
			rgb = list(webcolors.hex_to_rgb(d[attr]))
		rgb[color_dim] = int( rgb[color_dim]+value )
		color = webcolors.rgb_to_hex(rgb)
		d[attr] = color


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-c", "--csv", required=True,help="original csv")
	ap.add_argument("-s", "--svg", required=True,help="original svg")
	ap.add_argument("-cn", "--csv_new", required=True,help="new csv")
	ap.add_argument("-sn", "--svg_new", required=False,help="new svg")
	args = vars(ap.parse_args())

	csv_filename = args['csv']
	svg_filename = args['svg']
	csv_new_filename = args['csv_new']

	if args['svg_new'] is not None :
		svg_new_filename = args['svg_new']
	else :
		svg_new_filename = svg_filename.replace('.svg','-new.svg')

	with open(svg_filename) as fd:
	    svg = xmltodict.parse(fd.read())
	#print(xmltodict.unparse(doc, pretty=True))
	df = pd.read_csv(csv_filename).reset_index()#.reset_index()

	seqs = list(get_seqs(svg))
	if VERBOSE :
		print('#detected seqs:',len(seqs))
		print('\n'.join( str(x) for x in seqs ))

	rules = list()
	for seq in tqdm(seqs) :
		Y = np.array(seq.Y)
		rule = get_rule(df,Y)
		if rule is None:
			continue
		rule.paths = seq.paths
		rules.append(rule)

	if VERBOSE :
		print('#### detected rules:',len(rules))
		print('\n'.join( str(x) for x in rules ))

	df_new = pd.read_csv(csv_new_filename).reset_index()
	df_num = to_num(df)
	df_new_num = to_num(df_new)

	for col in df_num :
		if col not in df_new_num :
			df_new_num[col] = 0
	df_new_num = df_new_num[df_num.columns]
	svg_new = copy.deepcopy(svg)

	for rule in rules :
		X = df_new_num.values
		Y = X @ rule.a + rule.b
		for path,y in zip(rule.paths,Y):
			set_dict(svg_new,path,y)
		#print(X,Y,rule.paths)

	with open(svg_new_filename,'w') as f:
		f.write(xmltodict.unparse(svg_new,pretty=True))
