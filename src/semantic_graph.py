import matplotlib.pyplot as plt
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
                elif (filepath.endswith(".csv")):
                    continue
                else:
                    self.list_of_files.append(str(file))

    def  readfiles(self):
        for file in self.list_of_files:
            with open(file,"r",encoding="UTF-8") as f:
                print(file)
                self.read_files[file]=f.readlines()


class preprocessing:
    lists_is = ["newsgroups", "date", "path", "subject", "organization", "messageid", "lines", "gmt", "apr", "sender",
                "references", "article", "university", "time", "writes", "people", "nntppostinghost", "dont", "replyto",
                "mon", "im", "world", "distribution", "approved", "articleid", "california", "email", "compwindowsx",
                "recsporthockey", "list", "recmotorcycles", "socreligionchristian", "scimed", "aprathosrutgersedu",
                "christianaramisrutgersedu", "hedrickathosrutgersedu", "didnt", "day", "source", "xref", "found",
                "information", "sun", "provide", "cantaloupesrvcscmuedu", "read", "local", "called", "questions",
                "info"]
    totalvocab_tokenized = []
    # token_list= {Filename:[   list of all tokens]   }
    token_list = {}
    # termfrequency_file= {Filename: [term,frequency] }
    termfrequency_file = {}
    # termfrequency_overall= {[term,frequency]}
    termfrequency_overall = {}
    document_frequency = {}
    # inverse document frequency ={[term , idf value]}
    # idf value= log10(total documents/Number of documents term occured)
    idf = {}
    # term frequency and inverse document frequency(tf_idf) weight  ={ Filename: [ term , tf_idf value ] }
    # tf_idf value= term frequency * idf value of the term   (different for same terms in different documents)
    tf_idf = {}
    top_400 = {}
    normalized_tf = {}

    normalized_cftf = {}

    def __init__(self, read):
        self.read_files = read

    def tokenize_only(self, text):
        filtered_tokens = []
        stopwords = nltk.corpus.stopwords.words('english')
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        text = text.replace(".", "").replace("+", " ").replace("/", " ").replace(";", " ").replace("'s", "").replace(
            "-", "").replace("'", "").replace('"', " ").replace(',', ' ').replace(":", "").replace("@", "")
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                if re.search('[^\sx\s]', token):
                    if token not in stopwords:
                        filtered_tokens.append(token)
        return filtered_tokens

    def preprocess(self):

        for file, lines in self.read_files.items():
            strr = ""
            for words in lines:
                strr += words
            self.allwords_tokenized = self.tokenize_only(strr)
            self.totalvocab_tokenized.extend(self.allwords_tokenized)
            self.token_list[file] = []
            self.token_list[file].extend(self.allwords_tokenized)

        # iterates through tokens created and counts their frequency
        for filename, tff in self.token_list.items():
            self.termfrequency_file[filename] = {}
            wordcount = 0
            for word in tff:
                if word in self.termfrequency_file[filename]:
                    self.termfrequency_file[filename][word] += 1
                    wordcount += 1
                else:
                    self.termfrequency_file[filename][word] = 1
                    wordcount += 1
            #### total word count in a file stored in dict with key word_count
            self.termfrequency_file[filename]["word_count"] = wordcount

        # iterates through term frequency to count overall frequency
        for filename, tf_list in self.termfrequency_file.items():
            for term, frequency in tf_list.items():
                if term in self.termfrequency_overall:
                    self.termfrequency_overall[term] += frequency
                else:
                    self.termfrequency_overall[term] = frequency

        for term, frequency in self.termfrequency_overall.items():
            self.document_frequency[term] = 0
            for filename, tf_list in self.termfrequency_file.items():
                if term in tf_list:
                    self.document_frequency[term] += 1

        # calculates idf
        for term, frequency in self.document_frequency.items():
            i = 50 / (self.document_frequency[term])
            self.idf[term] = math.log10(i)

        '''
        #calculates tf_idf
        for term,frequency in self.document_frequency.items():
            self. tf_idf[term] =self.termfrequency_overall[term]*self.idf[term]
         '''

        self.normalized_tf = {}

        for filename, terms in self.termfrequency_file.items():
            self.normalized_tf[filename] = {}
            for term_name, term_freq in terms.items():
                self.normalized_tf[filename][term_name] = term_freq / (self.termfrequency_file[filename]["word_count"])

        for filename, terms in self.normalized_tf.items():
            self.tf_idf[filename] = {}
            for term_name, term_freq in terms.items():
                self.tf_idf[filename][term_name] = term_freq * self.idf[term_name]

        for term, frequency in self.document_frequency.items():
            self.normalized_cftf[term] = 0

        for filename, terms in self.normalized_tf.items():

            for term_name, term_freq in terms.items():
                self.normalized_cftf[term_name] += term_freq


class graph2word:
    graphs={}

    def __init__(self, termfrequency_file,token_list):

        for files,b in token_list.items():

            self.graphs[files] = nx.DiGraph()

            t_list=[]
            t_list=copy.deepcopy(token_list[files])

            prev = ""
            for words in b:

                if termfrequency_file[files][words] > 0:
                    if self.graphs[files].has_node(words):
                       self.graphs[files].node[words]['count']+=1
                    else:
                       self.graphs[files].add_node(words,count=1)
                    if self.graphs[files].has_edge(prev,words):
                        self.graphs[files][prev][words]['weight']+=1
                    else:
                        if prev is not "":
                            self.graphs[files].add_edge(prev,words,weight=1)
                    prev=words

    def getMCS(self, G_source, G_new):
        matching_graph = nx.Graph()

        for n1, n2, attr in G_new.edges(data=True):
            if G_source.has_edge(n1, n2):
                matching_graph.add_edge(n1, n2, weight=1)
        return matching_graph
        graphs = list(nx.connected_component_subgraphs(matching_graph))
        mcs_length = 0
        mcs_graph = nx.Graph()
        for i, graph in enumerate(graphs):

            if len(graph.nodes()) > mcs_length:
                mcs_length = len(graph.nodes())
                mcs_graph = graph

        return mcs_graph

    def get_MCS_GT(self,GT):
        for subdir, vec in GT.items():
            for subdir_1, vec_1 in GT.items():
                if (subdir==subdir_1):
                    for a in vec:
                        for b in vec_1:
                            if (a!=b):
                                max_subgraph =self.getMCS(self.graphs[a],self.graphs[b])
                                graph_pos = nx.circular_layout(max_subgraph, dim=2)
                                # nx.draw_networkx_nodes(max_subgraph, graph_pos, node_size=80, node_color='b', alpha=0.3)
                                # nx.draw_networkx_edges(max_subgraph, graph_pos, edge_color='grey', width=0.6, style="dashed")
                                # nx.draw_networkx_edge_labels(max_subgraph, graph_pos, font_size=5, font_color='black')
                                # nx.draw_networkx_labels(max_subgraph, graph_pos, font_size=8, font_color='black')
                                # plt.axis('off')
                                # plt.draw()
                                # plt.savefig("C:\\Users\\MUHAMMAD\\PycharmProjects\\fyp\\src\\pics\\" + str(a)+"--"+str(b) + ".png")
                                # plt.clf()
                                nx.write_pajek(max_subgraph, "C:\\Users\\MUHAMMAD\\PycharmProjects\\fyp\\src\\pajek\\" + str(a)+"--"+str(b) + ".net",encoding="UTF-8")

def tokenize_only( text):
    filtered_tokens = []
    stopwords = nltk.corpus.stopwords.words('english')
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    text = text.replace(".", "").replace("+", " ").replace("/", " ").replace(";", " ").replace("'s", "").replace("-",
                                                                                                                 "").replace(
        "'", "").replace('"', " ").replace(',', ' ').replace(":", "").replace("@", "")
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            if re.search('[^\sx\s]', token):
                if token not in stopwords:
                    filtered_tokens.append(token)
    return filtered_tokens
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


#Get dataset
a=Input("C:\\Users\\MUHAMMAD\\PycharmProjects\\fyp\\src")
a.readfiles()
b=preprocessing(a.read_files)
b.preprocess()

c=graph2word(b.termfrequency_file,b.token_list)

GT=getGT("C:\\Users\\MUHAMMAD\\PycharmProjects\\fyp\\src\\Doc50 GT")

c.get_MCS_GT(GT)
