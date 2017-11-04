
import os
import nltk
import re
import math
from symbol import term
import random
import numpy
import sys
import numpy as np
import copy
import networkx as nx

class Input:

    list_of_files=[]
    read_files={}

    def __init__(self,path):
        for subdir, dirs, files in os.walk(path):
           for file in files:
                filepath = subdir + os.sep + file
                if(filepath.endswith(".py")):
                    continue
                elif (filepath.endswith(".txt")):
                    continue
                elif(filepath.endswith(".csv"))    :
                    continue
                else:
                    self.list_of_files.append(str(file))

    def  readfiles(self):
        for file in self.list_of_files:
            with open(file,"r",encoding="UTF-8") as f:
                self.read_files[file]=f.readlines()


class preprocessing:
    token_list={}
    def __init__(self,read):
        self.read_files=read

    def tokenize_only(self,text):
        filtered_tokens = []
        stopwords = nltk.corpus.stopwords.words('english')
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        text = text.replace(".","").replace("+", " ").replace("/", " ").replace(";", " ").replace("'s", "").replace("-", "").replace( "'", "").replace('"', " ").replace(',', ' ').replace(":","").replace("@","")
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                if re.search('[^\sx\s]', token):
                    if token not in stopwords:
                        filtered_tokens.append(token)
        return filtered_tokens

    def preprocess(self):

        for file,lines in self.read_files.items():
            strr=""
            for words in lines:
                strr+=words
            allwords_tokenized = self.tokenize_only(strr)
            self.token_list[file]=[]
            self.token_list[file].extend(allwords_tokenized)


class hac:
    idf = {}
    no_ofclusters = 5
    clusters = {}
    matrix = nx.DiGraph

    # stores vectors
    def __init__(self, data,no_of_clulsters=5):
        self.idf = datatograph(data)
        self.no_of_clulsters=no_of_clulsters

    def iteration(self):
        self.matrix = nx.DiGraph()
        for keys in self.idf.keys():

            for ref in self.idf.keys():
                if(ref != keys):
                    self.matrix.add_edge(keys,ref,weight=gsimm(self.idf[keys],self.idf[ref]))
        maxi = -sys.maxsize
        file1 = str
        file2 = str

        while(len(self.matrix.nodes())>self.no_ofclusters):
            maxi = -sys.maxsize
            file1 = str
            file2 = str

            for f2,f3 in self.matrix.edges():

                if (self.matrix[f2][f3]['weight']>maxi):
                    maxi=self.matrix[f2][f3]['weight']
                    file1=f2
                    file2=f3
            for node in self.matrix.nodes():
                if node!=file1 and node!=file2:
                    w= (self.matrix[node][file1]['weight']+self.matrix[node][file2]['weight'])/2
                    self.matrix.add_edge(node,file1+" "+file2,weight=w)
                    self.matrix.add_edge(file1 + " " + file2,node , weight=w)
            self.matrix.remove_node(file1)
            self.matrix.remove_node(file2)
        whatever={}
        i=0
        for nodes,data in self.matrix.nodes(data=True):
            node=str(nodes)
            node=node.split()
            whatever[i]=node
            i+=1
        return whatever

def datatograph(token_list):
    glist = {}
    for files, b in token_list.items():

        glist[files] = nx.DiGraph()

        t_list = []
        t_list = copy.deepcopy(token_list[files])
        prev = ""
        for words in b:
            if glist[files].has_node(words):
                glist[files].node[words]['count'] += 1
            else:
                glist[files].add_node(words, count=1)
            if glist[files].has_edge(prev, words):
                glist[files][prev][words]['weight'] += 1
            else:
                if prev is not "":
                    glist[files].add_edge(prev, words, weight=1)
            prev = words
    return glist



def gsimm(a,b):
    dif = nx.DiGraph()
    R = len(set.intersection(set(a.nodes()),b.nodes()))
    e_a = set(a.edges_iter())
    e_b = set(b.edges_iter())
    E = len(set.intersection(e_a,e_b))
    a=set(a.nodes())

    b=set(b.nodes())
    #return (len(a)+len(b))
    return ((R/(len(b)+len(a)-R))+ (E/(len(e_a)+len(e_b)-E)))/2


#gets ground truth files and reads them
def getGT(path):
    GT={}
    for subdir, dirs, files in os.walk(path):
        filepath = subdir
        GT[filepath]=[]
    for subdir, dirs, files in os.walk(path):

        for file in files:
            filepath = subdir
            GT[filepath].append(str(file))


    return GT


def purity(result,GT,totalfiles=50,no_of_clusters=5):
    way={}
    result=copy.deepcopy(result)
    GT=copy.deepcopy(GT)

    for i in range(0,no_of_clusters):
        ind=""
        sub=""
        length = -sys.maxsize
        #iterates through results and ground truth and finds the one with largest intersection
        for index, file in result.items():



            for subdir, vec in GT.items():
                if(length<=len(list(set(file).intersection(vec)))):
                    length=len(list(set(file).intersection(vec)))
                    ind=index
                    sub=subdir
        way[i]=length
        #Prints things
        #pops the clusters and ground truth group that has been used so it cannot be used again
        #print(i)
        print("Result==  "+str(result[ind]))
        print("Ground truth=== "+str(GT[sub]))
        result.pop(ind)
        GT.pop(sub)
    #finds purity
    true_values=0
    for i,b in way.items():
        true_values+=b

    return (true_values/totalfiles)



#Get dataset
a=Input("C:\\Users\\muhammad\\PycharmProjects\\fyp\\src")
a.readfiles()

#preprocess dataset
b=preprocessing(a.read_files)
b.preprocess()

GT = getGT("C:\\Users\\muhammad\\PycharmProjects\\fyp\\src\\Doc50 GT")


h=hac(b.token_list)
print("HAC Graph Purity")
print(purity(h.iteration(),GT,totalfiles=19,no_of_clusters=4))

