import sys
from sklearn.datasets import load_svmlight_file

from sklearn.ensemble import RandomForestClassifier

from time import time

import numpy as np


def dumptree(atree, fn):
	from sklearn import tree
	f = open(fn,"w")
	tree.export_graphviz(atree,out_file=f)
	f.close()

# def main():
fn = sys.argv[1]
X,Y = load_svmlight_file(fn)

ntr = int(float(sys.argv[2]))
ncore = int(float(sys.argv[3]))

rf_parameters = {
	"n_estimators": ntr,
	"n_jobs": ncore
}
clf = RandomForestClassifier(**rf_parameters)
X = X.toarray()

print clf

print "Starting Training"
t0 = time()
clf.fit(X, Y)
train_time = time() - t0
print "Training on %s took %s"%(fn, train_time)
print "Total training time (seconds): %s"%(train_time)

fn2 = sys.argv[4]
X2,Y2 = load_svmlight_file(fn2)
X2 = X2.toarray()
score = clf.score(X2, Y2)
fn3 = sys.argv[5]
Y2P = clf.predict(X2)
C = np.array([Y2P,Y2]).T
np.savetxt(fn3, C, delimiter=",")
