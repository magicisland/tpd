from __future__ import division
from collections import Counter
from nltk.metrics import ConfusionMatrix


"""
ref    = 'DET NN VB DET JJ JJ NN IN DET NN'.split()
tagged = 'DET VB VB DET NN JJ NN IN DET NN'.split()
cm = ConfusionMatrix(ref, tagged)
labels = set('DET NN VB IN JJ'.split())

true_positives = Counter()
false_negatives = Counter()
false_positives = Counter()

for i in labels:
    for j in labels:
        if i == j:
            true_positives[i] += cm[i,j]
        else:
            false_negatives[i] += cm[i,j]
            false_positives[j] += cm[i,j]

print "TP:", sum(true_positives.values()), true_positives
print "FN:", sum(false_negatives.values()), false_negatives
print "FP:", sum(false_positives.values()), false_positives
"""

ref    = 'DET NN VB DET JJ JJ NN  IN NN VB DET NN'.split()
tagged = 'DET VB VB DET VB VB DET NN JJ JJ NN  IN'.split()
cm = ConfusionMatrix(ref, tagged)
labels_ref = set(ref)
labels_tagged=set(tagged)
print cm


true_positives = {}
false_negatives = Counter()
false_positives = Counter()

print ref
print tagged

set_ref=[]
set_tagged=[]

for row in ref:
	if row not in set_ref:
		set_ref.append(row)
for col in tagged:
	if col not in set_tagged:
		set_tagged.append(col)

print set_ref
print set_tagged

def aragua(i,data):
	
	total=sum(data[i].values())
	if total==0:
		total+=1
	probabilidades=[]

	for value in data[i]:
		freq=data[i].get(value)
		prob=float(freq/total)
		probabilidades.append((value+" : "+str(round(prob,2))))
	return probabilidades

for i in set_ref:
    for j in set_tagged:
    	if i not in true_positives:
        	true_positives[i]=Counter()
       		true_positives[i][j]+=cm[i,j]
      	else:
   			true_positives[i][j]+=cm[i,j]

for i in true_positives:
	print i,Counter(true_positives[i]),sum(true_positives[i].values()),aragua(i,true_positives)



