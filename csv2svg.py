import sys
import xmltodict
import numpy as np
import pandas as pd
import copy
from sklearn.linear_model import LinearRegression, Lasso
from collections import OrderedDict as dict
NUMERICS = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
VERBOSE = True

if len(sys.argv) > 1 :
	csv_filename = sys.argv[1]
	csv_new_filename = sys.argv[2]
	svg_filename = sys.argv[3]
	svg_new_filename = svg_filename.replace('.svg','-new.svg')
else :
	csv_filename = 'test/simple.csv'
	csv_new_filename = 'test/simple-new.csv'
	svg_filename = 'test/simple-bubbles.svg'
	svg_new_filename = 'test/simple-bubbles-new.svg'

with open(svg_filename) as fd:
    svg = xmltodict.parse(fd.read())
#print(xmltodict.unparse(doc, pretty=True))
df = pd.read_csv(csv_filename).reset_index()#.reset_index()
df_num = df.select_dtypes(include=NUMERICS)

###################################################%%
### get all sequences in svg
###################################################%%

class Sequence(object) :
	def __init__(self,paths=None,Y=None) :
		if Y is None:
			Y = list()
		if paths is None:
			paths = list()
		self.Y = Y
		self.paths = paths
	def __str__(self) :
		return str(self.paths[0])+' -> '+str(self.Y)


def get_seqs(node,path=list()):
	if type(node) == list:
		for i,node_ in enumerate(node):
			yield from get_seqs(node_,list(path)+[i])
	elif type(node) == dict:
		for key in node:
			if key not in {'rect','ellipse'} :
				yield from get_seqs(node[key],list(path)+[key])
			elif type(node[key]) == list:
				if key == 'rect' :
					attrs = ['@x','@y','@height','@width']
				elif key == 'ellipse' :
					attrs = ['@cx','@cy','@rx','@ry']
				for att in attrs :
					seq = Sequence()
					for i,node_ in enumerate(node[key]):
						seq.paths.append(tuple(list(path)+[key,i,att]))
						seq.Y.append(float(node_[att]))
					yield seq

seqs = list(get_seqs(svg))
if VERBOSE :
	print('#detected seqs:',len(seqs))
	print('\n'.join( str(x) for x in seqs ))

###################################################%%
### match sequences with possible rules
###################################################%%

class ScaleRule(Sequence) :
	def __init__(self,paths=None,Y=None,cols=None,a=None,b=None):
		super(ScaleRule, self).__init__(paths,Y)
		self.cols = cols
		self.a = a
		self.b = b
	def __str__(self):
		return str((self.cols,list(self.a),self.b,self.paths[0]))

def get_rule(df,Y,eps=0.001,alpha=0.000000000001) :
	if len(df) != len(Y) or min(Y) == max(Y) :
		return None
	mod = LinearRegression()
	#mod = Lasso(alpha=alpha)
	X = df_num.values
	Y_pred = mod.fit(X,Y).predict(X)
	score = np.abs( Y - Y_pred ).mean()
	b = mod.intercept_
	if b < -eps or score > eps:
		return None
	a = mod.coef_.astype(int)
	b = int(b)
	a[ np.abs(a) < eps ] = 0
	rule = ScaleRule(cols=df_num.columns.tolist(),a=a,b=b)
	return rule

rules = list()
for seq in seqs:
	Y = np.array(seq.Y)
	rule = get_rule(df,Y)
	if rule is None:
		continue
	rule.paths = seq.paths
	rules.append(rule)

if VERBOSE :
	print('#### detected rules:',len(rules))
	print('\n'.join( str(x) for x in rules ))


###################################################%%
### apply rules to new dataset
###################################################%%

df_new = pd.read_csv(csv_new_filename).reset_index()
df_new_num = df_new.select_dtypes(include=NUMERICS)
svg_new = copy.deepcopy(svg)

def set_dict(d,path,value) :
	for key in path[:-1] :
		d = d[key]
	d[path[-1]] = value

for rule in rules:
	X = df_new_num.values
	Y = X @ rule.a + rule.b
	for path,y in zip(rule.paths,Y):
		set_dict(svg_new,path,y)
	#print(X,Y,rule.paths)

with open(svg_new_filename,'w') as f:
	f.write(xmltodict.unparse(svg_new,pretty=True))
