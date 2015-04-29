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

		
		train_sents=corpus[:size]
		test_sents=corpus[size:]

		"""
		for i in range(0,len(corpus)):
			if i%2==0:
				train_sents.append(corpus[i])
			else:
				test_sents.append(corpus[i])
		"""	

		

		
		sleep(7.0)

		self.tt=[]
		try:
			for  ts in test_sents:
				#print "TS:",ts
				#print "EN conlltags:",conlltags2tree(ts)
				
				self.tt.append(conlltags2tree(ts))
		
		except Exception,e:
			print "Exception:",str(e)
			pdb.set_trace()

		print "Porcentaje del corpus de entrenamiento:",training_portion
		print "Porcentaje del corpus para testing:",1-training_portion

		
		self.chunker=Chunker(train_sents,chunkers_path,file_extension,feature_extractor)

	
	def createModel(self):
				
		print self.chunker.evaluate(self.tt)
		return self.chunker.save()



feature_extractor=locals()['ContextFeatureExtractor']()
generador=ChunkerModelGenerator("/home/marce/code/virtual_envs/nerit/tagger_models/",
	"/home/marce/code/virtual_envs/nerit/corpora/tweet_corpus/",['corregido.tcs'], ('TIMEP','VERBP','WEBP','LOCATION','O','EVENT','NOUNP','WEBP','ADVBP'),
	0.5,".chnk",feature_extractor)
generador.createModel()


