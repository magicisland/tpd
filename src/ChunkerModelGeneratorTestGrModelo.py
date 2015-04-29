#!/usr/bin/env python
# -*- coding: UTF-8 -*

from __future__ import division
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
from time import sleep
from collections import Counter
from nltk.metrics import ConfusionMatrix

#**********************************************************************************************************************************************

from datasource.TwitterConnection import *
from datasource.FileConnection import *
from datasource.DataLineConnection import *
from NeritFacade import *
from Printer import *
import codecs
import pickle
import os
import sys
from filters.Filter import StringFilter
import os.path
from time import sleep

from grammars_fix import * #aqui tengo las gramaticas para corregir falsos negativos de las direcciones
import numpy as np 


"""
	Este modulo es la instanciacion del framework. 

"""

def usage():
	print " "
	print "[Opción] [argumentos]"
	print "-cpt 	Crear post tagger según archivo de configuración"
	print "-cct     Crear chunk tagger seǵun archivo de configuración"
	print "argumentos:    ruta del archivo de configuración para los taggers."
	print "-tc 		Procesar desde Twitter"
	print "argumentos: lista de hashtags. Ej 'tránsito, choque' "
	print "-lc 		Procesar desde archivo"
	print " "



def strToList(string_data):

	data_list=[]
	if isinstance(string_data,str):
		data_list=string_data.split()
	return data_list	


def createTwitterConnection(nerit,hashtags=[]):

	
	timeLine=True
	if hashtags:
		timeLine=False
	connection=nerit.getTwitterConnection(hashtags,timeLine)
	return connection


def getAdHocStrategy(grammar_location=locations,grammar_clean=clean):

	# utilizar las gramáticas de grammars_fix.py para corregir falsos negativos.

	locations=PostPatternStrategy(grammar_location) # gramaticas para direcciones
	
		
	iobFixer=IOBFixerStrategy()						# """" iobFixer me permite acomodar el Arbol de Chunks luego de recortar por detectar frases.""""

	fixer=SequentialStrategy(locations,iobFixer)	# armar una estrategia para corregir direcciones usando la gramatica "locations"

	clean=PostPatternStrategy(grammar_clean)      # aplicar unos ajustes mas...
	fixer=SequentialStrategy(fixer,clean)
	fixer=SequentialStrategy(fixer,iobFixer)

	#decorator=WrappedStrategyTagger()
	#decorator.set_strategy(fixer)
	return fixer


def createPostStage(nerit,post_model):

	decoratorTagger=None#nerit.getDecoratorTagger()
	posTaggerStage=nerit.getTaggerStage(model_path=post_model,decorator=decoratorTagger)
	if os.path.exists(post_model):
		posTaggerStage=nerit.getTaggerStage(post_model,'text','tagged',decoratorTagger)

	return posTaggerStage
	

def createChunkStage(nerit,chunk_model):

	# combinar extracciones por metodo probabilistico y strategy de gramaticas para corregir los falsos negativos.

	decorator=nerit.getChunkerDecorator()
	chunkerStage=nerit.getChunkerStage(decorator=decorator) # hasta aqui, sólo va a utilizar puramente gramaticas para extraer direcciones y 
	# le aniadimos un poco mas de gramaticas para regifinar los sintagmas que buscamos. Recordar, que asi como esta primero va a usar la
	# gramatica "chunks"
	
	if os.path.exists(chunk_model):
		# pero si hay un modelo entrenado, lo utilizamos.
		
		decorator=getAdHocDecorator() # y también corregimos algunos errores con una estrategias de ayuda.( o podria utilizar las mismas que x regexp comentamos....)
		chunkerStage=nerit.getChunkerModelStage(model_path=chunk_model,decorator=decorator)
		print "\nUtilizando el modelo entrenado para extraer chunks:",chunk_model
		sleep(1.0)
	return chunkerStage



def createChunkStage_(nerit,chunk_model):

	# este es el que combina todo y no va a estar incluido en la solucion, porque combinar puramente expresiones ( que detectan direcciones desde
	# chunks básicos ( sintagmas nominales,verbales,etc) rompe la solucion anterior para luego aplicar una estrategia y refinar los sintagmas
	# normales en busca de direcciones y eventos. 
	
	chunkerStage=None
	if os.path.exists(chunk_model):
		chunkerStage=nerit.getChunkerWrappedStage(model_path=chunk_model,strategy=nerit.getChunkerStrategy())
		print "\nUtilizando el modelo entrenado para extraer chunks:",chunk_model
		sleep(1.0)

	return chunkerStage




def createPipeline(nerit,connection,tokenizers,abbreviations,post_model,chunk_model,save_to,strFilter,corpus_size,final_stage=None):

	# crear el pipeline con todas las etapas...
	
	try:
		tokenizerStage=nerit.getTokenizerStage(tokenizers,abbreviations)
		posTaggerStage=createPostStage(nerit,post_model)
		
		chunkerStage=createChunkStage(nerit,chunk_model)
		persistenceStage=nerit.getPersistenceStage(StringFilter(strFilter),save_to,corpus_size)
		
		nerit.add_finalStage(final_stage)
		
		pipeline=nerit.createPipeline([tokenizerStage,posTaggerStage,chunkerStage,persistenceStage])
		
		connection.addObserver(pipeline)
		connection.listen()

	except Exception,e:
		#print str(e)
		#pdb.set_trace()	
		pass





#*********************************************************************************************************************************************


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




def eval(gold,test):

	presicion=0
	n=0
	tp=fp=0
	for i in range(0,len(gold)):
		gold_sent=gold[i]
		test_sent=test[i]
		n+=len(gold[i])
		for j in range(0,len(gold[i])):
			#pdb.set_trace()
			gold_w,gold_p,gold_iob=gold_sent[j]
			test_w,test_p,test_iob=test_sent[j]
			if gold_iob==test_iob:
				tp+=1
			else:
				fp+=1
			
	presicion=tp/(tp+fp)
	print "presicion en IOBS:",presicion

def estadistica_global(cm,label_row,label_col):

	true_positives=Counter()
	false_negatives=Counter()

	for row in label_row:
		for col in label_col:
			if row==col:
				true_positives[row]+=cm[row,col]
			else:
				false_negatives[row]+=cm[row,col]

	total=sum(true_positives.values())+sum(false_negatives.values())
	tp=sum(true_positives.values())
	fn=sum(false_negatives.values())
	
	prob_tp=float(tp/total)
	prob_fn=float(fn/total)
	
	print "Cantidad de True positives:",tp," porcentaje:_",round(prob_tp,2)
	print "Cantidad de FAlse negatives:",fn," porcentaje:",round(prob_fn,2)

def aragua(i,data):
	
	total=sum(data[i].values())
	if total==0:
		total+=1
	probabilidades=[]
	probs_puras=[]
	tags=[]
	for value in data[i]:
		freq=data[i].get(value)
		prob=float(freq/total)
		probabilidades.append((value+" : "+str(round(prob,2))))
		probs_puras.append(round(prob,2))
		tags.append(value)
		
	return probabilidades,round(np.var(probs_puras),2),tags

	
class Chunker(nltk.ChunkParserI):

	def __init__(self,train_sents,chunkers_path,extension_file,feature_extractor):
		
		nerit=NeritFacade()
		#self.strategy=getAdHocStrategy()
		self.strategy=nerit.getChunkerStrategy()
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
		conlltags = [(w,t,c) for ((w,t),c) in tagged_sent]
		conlltags=self.strategy.fix(conlltags)
		return nltk.chunk.conlltags2tree(conlltags)


def distinto_y_tiene (x=['EVENT','NOUNP','WEBP','TIMEP','O','ADVBP','PREPP'],lista=[]):

	
	for i in x:
		for j in lista:
			if i in j or j in i:
				return True
	return False


class ChunkerModelGenerator(object):

	def __init__(self,chunkers_path,corpus_path,files,phrases,training_portion,file_extension,feature_extractor):
	
		path='/home/marce/tp_diseño/nerit/corpora/tweet_corpus/'
		files=['']
		conllreader = ConllChunkCorpusReader(path,files,phrases)
		gold=conllreader.iob_sents()


		conllreader=ConllChunkCorpusReader(path,files,phrases)
		test=conllreader.iob_sents()
		
		print "Cantidad de tweets en corpus gold:",len(gold)
		print "Cantidadd de tweets en corpus test:",len(test)

		eval(gold,test)
	

def get_texto(sent):

	text=[]
	for w,t,c in sent:
		text.append(w)
	string_text=' '.join(text).encode('UTF-8')


def test(gold,test):

	# gold: es el archivo posta.



feature_extractor=ContextFeatureExtractor()
generador=ChunkerModelGenerator("/home/marce/code/virtual_envs/nerit/tagger_models/",
	"/home/marce/code/virtual_envs/nerit/corpora/tweet_corpus/",['gold.tcs'], ('TIMEP','VERBP','WEBP','LOCATION','O','EVENT','NOUNP','WEBP','ADVBP'),
	0.8,".chnk",feature_extractor)


