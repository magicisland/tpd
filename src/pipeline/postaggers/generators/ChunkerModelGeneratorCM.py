#!/usr/bin/env python
# -*- coding: UTF-8 -*

import nltk.chunk
from nltk.tag import ClassifierBasedTagger
from nltk.classify import *
from nltk.tag import *
import nltk.chunk, itertools
from nltk.corpus import conll2000
from nltk.corpus.reader import ConllChunkCorpusReader
import random
from nltk.corpus import names
from nltk.chunk import RegexpParser
import pdb
from nltk.chunk.util import *
from time import sleep
import os.path
from os import listdir
import pickle
from nltk.metrics import ConfusionMatrix
from time import sleep
from collections import Counter

class ContextFeatureExtractor(object):

	def __init__(self):
		super(ContextFeatureExtractor,self).__init__()

	def get_features(self,tokens,index,history):

		prevword,prevpos,previob=('<START>',)*3
		actual_word,actual_pos=tokens[index]
		nextword,nextpos,nextiob=('<END>',)*3

		word=actual_word
		pos=actual_pos

		feats={}
		
		if index>0:

			prev_word,prev_pos=tokens[index-1]
			prev_iob=history[index-1]

			prevpos=prev_pos
			previob=prev_iob
			prevword=prev_word

		if index < len(tokens)-1:

			next_word,next_pos=tokens[index+1]
			
			nextpos=next_pos
			nextword=next_word


		feats['prevword']=prevword
		feats['prevpos']=prevpos
		feats['previob']=previob
		feats['word']=word
		feats['pos']=pos
		feats['nextword']=nextword
		feats['nextpos']=nextpos
		feats['nextiob']=nextiob
		#print "FEATS:",feats
		return feats


class Classifier(nltk.TaggerI):

	def __init__(self,train_sents,chunkers_path,extension_file,feature_extractor):
		
		self.chunkers_path=chunkers_path
		self.extension_file=extension_file
		self.feature_extractor=feature_extractor

		train_set=[]
		for sent in train_sents:
			
			history=[]
			untagged_sent=nltk.tag.untag(sent)
			for i, (word, tag) in enumerate(sent):
				features=feature_extractor.get_features(untagged_sent,i,history)
				if features:
					train_set.append((features,tag)) # voy a formar un conjunto de entrenamiento unicamente con (word,pos,tag) de "web y tiempos"
					history.append(tag)
					

			
		self.classifier=nltk.NaiveBayesClassifier.train(train_set)
		#print self.classifier.labels()
		
	
	def save(self,chunker_name):
		if self.classifier:
			if not chunker_name:
				chunker_name=self.getDefaultName()
			f = open(chunker_name,'wb')
			pickle.dump(self, f)
			f.close()
		print ("Guardando el chunker :" + chunker_name)
		return chunker_name

	def getDefaultName(self):

		fname=self.chunkers_path+"chunker"
		fname=fname+str(len(listdir(self.chunkers_path)))+self.extension_file
		return fname
		
	def tag(self,sentence):

		history=[]
		for i,word in enumerate(sentence):
			features=self.feature_extractor.get_features(sentence,i,history)
			tag=self.classifier.classify(features)
			history.append(tag)
		tagged=zip(sentence,history)
		return tagged


class Chunker(nltk.ChunkParserI):

	def __init__(self,train_sents,chunkers_path,extension_file,feature_extractor):
		
	
		tagged_sents=[]
		for sent in train_sents:
			tagged_sent=[]
			for word,post,chunk in sent:
				#print "WTC:",word,post,chunk
				tagged_sent.append(((word,post),chunk))
			tagged_sents.append(tagged_sent)	
		
		self.tagger=Classifier(tagged_sents,chunkers_path,extension_file,feature_extractor)

	def save(self,chunker_name=None):
		return self.tagger.save(chunker_name)

	def parse(self,sentence):

		#print "Sentence:",sentence
		tagged_sent=self.tagger.tag(sentence)
		#print "Taggeada sentence:",tagged_sent
		conlltags = [(w,t,c) for ((w,t),c) in tagged_sent]
		return nltk.chunk.conlltags2tree(conlltags)


class ChunkerModelGenerator(object):


	
	def __init__(self,chunkers_path,corpus_path,files,phrases,training_portion,file_extension,feature_extractor):

		
		conllreader = ConllChunkCorpusReader(corpus_path,files,phrases)
		corpus=conllreader.iob_sents()
		print "Cantidad de sentencias del corpus:",len(corpus)
		print "training_portion:",training_portion
		train_sents=[]
		test_sents=[]

		size=float(len(corpus)*training_portion)
		size=int(size)
		print "Cantida de sentencias de entrenamiento:",size
		
		gold_sents=corpus
		train_sents=corpus[:size]
		test_full=corpus[size:]
		gold_test_sents=corpus[size:]
		print "Cantidad total de sentencias de testing:",len(test_full)
		print "Cantidad de sentencias de testing para Matríz de Confusión:",len(gold_test_sents)


		self.chunker=Chunker(train_sents,chunkers_path,file_extension,feature_extractor)

		Itest_sents=[]

		for sent in gold_test_sents:
			#print "Etiquetada posta:",sent
			its=[]
			for w,t,c in sent:
				its.append((w,t))
			Itest_sents.append(its)

		
		gold=[]
		for sent in gold_test_sents:
			for w,t,c in sent:
				#if "IN" in t or "CD" in t or "NN" in t:
				gold.append(w+"_"+t+"_"+c)

		tagged=[]
		for ts in Itest_sents:
			#print "Etiquetada de prueba:",tree2conlltags(self.chunker.parse(ts))
			for w,t,c in tree2conlltags(self.chunker.parse(ts)):
				#if "IN" in t or "CD" in t or "NN" in t:
				tagged.append(w+"_"+t+"_"+c)

		gold_row=(' '.join(gold)).split()
		test_row=(' '.join(tagged)).split()

		print "Longitud de las listas de palabras:",len(gold),len(tagged)
	

		cm=nltk.ConfusionMatrix(gold_row,test_row)
		
		#print(cm.pp(sort_by_count=True, show_percents=True))
		
		labels=set(gold_row)

	

		true_positives=Counter()
		false_negatives=Counter()
		false_positives=Counter()
		
		for i in labels:
		    for j in labels:
		    	if i == j:
        			true_positives[i]+= cm[i,j]
        		else:
        			false_negatives[i]+= cm[i,j]
        			false_positives[j]+= cm[i,j]

		#print "TP:", sum(true_positives.values()), true_positives
		#print "FN:", sum(false_negatives.values()), false_negatives
		#print "FP:", sum(false_positives.values()), false_positives

		print "------------------------------------------------------------"
		for i in labels:
			if "LOCATION" in i:
				if get_false_negatives(cm,i,labels) > 0:
					print "-------------------------------------------------_"

	def createModel(self):
				
		print tree2conlltags(self.chunker.evaluate(self.tt))
		

def get_false_negatives(confusion_matrix,label,labels):

	
	# this method retrieve the label's false negatives
	fn=0
	for j in labels:
		if confusion_matrix[label,j] !=0 and j!=label:
			fn+=1	
			char_j=j
			char_j=char_j.encode('UTF-8')
			print ("Para:",label,": es:",char_j," ",confusion_matrix[label,j]) 
	return fn






feature_extractor=locals()['ContextFeatureExtractor']()
generador=ChunkerModelGenerator("/home/marce/code/virtual_envs/nerit/tagger_models/",
	"/home/marce/code/virtual_envs/nerit/corpora/tweet_corpus/",['gold.1.tcs'], ('TIMEP','VERBP','WEBP','LOCATION','O','EVENT','NOUNP','WEBP','ADVBP'),
	0.5,".chnk",feature_extractor)
#generador.createModel()


