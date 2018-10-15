# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec,KeyedVectors
from nltk import word_tokenize
from pyemd import emd
import pandas as pd
import numpy as np
import codecs
import jieba
from jieba import posseg
import re
import time
def LogInfo(stri):
    print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'  '+stri)
    
def preprocess_data_en(stopwords,doc):
    '''
    Function: preprocess data in Chinese including cleaning, tokenzing...
    Input: document string
    Output: list of words
    '''     
    doc = doc.lower()
    doc = word_tokenize(doc)
    doc = [word for word in doc if word not in set(stopwords)]
    doc = [word for word in doc if word.isalpha()]
    return doc

def preprocess_data_cn(stopwords,doc):
    '''
    Function: preprocess data in Chinese including cleaning, tokenzing...
    Input: 
        stopwords: Chinese stopwords list
        doc: document string
    Output: list of words
    '''       
    # clean data
    doc = re.sub(u"[^\u4E00-\u9FFF]", "", doc) # delete all non-chinese characters
    doc = re.sub(u"[儿]", "", doc) # delete 儿
    # tokenize and move stopwords 
#     doc = [word for word in jieba.cut(doc) if word not in set(stopwords)]  
    words = []
    pos = ['zg','e','y','o','ul','ud','uj','z'] # 定义需要过滤的词性
    # zg:哦 e:嗯 y:啦 o:哈哈 ul:了 r:他，你，哪儿，哪里 ug:过 z:咋啦
    seg = jieba.posseg.cut(doc)  # 分词
    for i in seg:   
        if i.flag not in pos and i.word not in stopwords :  # 去停用词 + 词性筛选
            words.append(i.word)            
    return words

def filter_words(vocab,doc):
    '''
    Function: filter words which are not contained in the vocab
    Input:
        vocab: list of words that have word2vec representation
        doc: list of words in a document
    Output:
        list of filtered words
    '''
    return [word for word in doc if word in vocab]

def f(x):
    if x<0.0: return 0.0
    else: return x
    
def handle_sim(x):  
    return 1.0-np.vectorize(f)(x)

def regularize_sim(sims):
    '''
    Function: replace illegal similarity value -1 with mean value
    Input: list of similarity of document pairs
    Output: regularized list of similarity 
    '''
    sim_mean = np.mean([sim for sim in sims if sim!=-1])
    r_sims = []
    errors = 0
    for sim in sims:
        if sim==-1:
            r_sims.append(sim_mean)
            errors += 1
        else:
            r_sims.append(sim)
#     LogInfo('Regularize: '+str(errors))
    return r_sims

def load_word2vec(model_path):
    model = dict()
    for line in open(model_path,encoding='utf-8'):
        l = line.strip().split()    
        st=' '.join(l[:-300]).lower()   
        model[st]=list(map(float,l[-300:]))
  
    num_keys=len(model)
   
    return model


def wmd_sim(lang,docs1,docs2):
    '''
    Function:
        calculate similarity of document pairs 
    Input: 
        lang: text language-Chinese for 'cn'/ English for 'en'
        docs1:  document strings list1
        docs2: document strings list2
    Output:
        similarity list of docs1 and docs2 pairs: value ranges from 0 to 1; 
                  
    '''
    # check if the number of documents matched
    assert len(docs1)==len(docs2) ,'Documents number is not matched!'
    assert len(docs1)!=0,'Documents list1 is null'
    assert len(docs2)!=0,'Documents list2 is null'
    assert lang=='cn' or lang=='en', 'Language setting is wrong'
    # change setting according to text language 
    if lang=='cn':
        model_path = '../model/cn.cbow.bin'
        stopwords_path = 'chinese_stopwords.txt'
        preprocess_data = preprocess_data_cn
    elif lang=='en':
        model_path = '../model/GoogleNews-vectors-negative300.bin'
        stopwords_path = 'english_stopwords.txt'
        preprocess_data = preprocess_data_en
    # load word2vec model  
    LogInfo('Load word2vec model...')
#     model = load_word2vec('../model/sgns.baidubaike.bigram-char')
#     vocab = list(model.keys())
    model = KeyedVectors.load_word2vec_format(model_path,binary=True,unicode_errors='ignore')
    vocab = model.vocab

    # preprocess data
    stopwords= set(w.strip() for w in codecs.open(stopwords_path, 'r',encoding='utf-8').readlines())
    sims = []
    LogInfo('Calculating similarity...')
    for i in range(len(docs1)):        
        p1 = preprocess_data(stopwords,docs1[i])
        p2 = preprocess_data(stopwords,docs2[i])
        p1 = filter_words(vocab,p1)
        p2 = filter_words(vocab,p2)
        if len(p1)==0 or len(p2)==0:
            # if any filtered document is null, return -1 
            sim = -1
        else:
            p1 = ' '.join(p1)
            p2 = ' '.join(p2)
            vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b', stop_words=None)
            v1,v2 = vectorizer.fit_transform([p1,p2])
            # pyemd needs double precision input
            v1 = v1.toarray().ravel().astype(np.double)
            v2 = v2.toarray().ravel().astype(np.double)
            # transform word count to frequency [0,1]
            v1 /= v1.sum()
            v2 /= v2.sum()
            # obtain word2vec representations 
            W = [model[word] for word in vectorizer.get_feature_names()]
            # calculate distance matrix (distance = 1-cosine similarity) [0,1]
            D = handle_sim(cosine_similarity(W)).astype(np.double)         
            # calculate minimal distance using EMD algorithm
            min_distance = emd(v1,v2,D)
            # calculate similarity (similarity = 1-min_distance)
            sim = 1-min_distance
        
        sims.append(sim)
    # regularize similarity: replace -1 with average similarity
    rsims = regularize_sim(sims) 
    # 只保留小数点后四位
    rsims = [round(sim,4) for sim in rsims]
    return rsims

def compute_ser(sims):
    '''
    Function: compute SER(semantic error rate) according to the document similarity
    Input: 
        sims: list of document similarity
    Output:
        sers: list of document SER
    '''
    sers = [round(1.0-sim,4) for sim in sims]
    return sers

def example():
    # English text example
    docs1 = ['man sitting using tool at a table in his home.',
                 'vegetable is being sliced.',
                'a speaker presents some products']
    docs2 = ['The president comes to China',
                'someone is slicing a tomato with a knife on a cutting board.',
                'the speaker is introducing the new products on a fair.']
    # calculate similarity
    sims = wmd_sim('en',docs1,docs2)
    # calculate SER
    sers = compute_ser(sims)
    # print result
    for i in range(len(sims)):
        print(docs1[i])
        print(docs2[i])
        print('Similarity: %.4f' %sims[i])
        print('SER: %.4f' %sers[i])
        
    # Chinese text example
    docs1 = ['时间太晚不得就算了', 
            '他整天愁眉苦脸',
             '学无止境'] 
             
    docs2 = ['此间贷款不得就算啦', 
            '他和朋友去逛街',
             '学海无涯，天道酬勤']
    # calculate similarity
    sims = wmd_sim('cn',docs1,docs2)
    # calculate SER
    sers = compute_ser(sims)
    # print result
    for i in range(len(sims)):
        print(docs1[i])
        print(docs2[i])
        print('Similarity: %.4f' %sims[i])
        print('SER: %.4f' %sers[i])
        
def main_cn():
    corpus = ['baidu_003_02','weixin_003_02','ifly_003_02',
              'baidu_008','weixin_008','ifly_008',
                'baidu_004','weixin_004', 'ifly_004',
                'baidu_006_01','weixin_006_01', 'ifly_006_01',           
                'baidu_004_02','weixin_004_02','ifly_004_02',
               'baidu_rePunct_huiting','weixin_rePunct_huiting', 'ifly_rePunct_huiting']

    for c in corpus:
        LogInfo(c+' start')     
        # read data
        data = pd.read_csv('../data/'+c+'.csv')
        docs1 = data.REF.values
        docs2 = data.HYP.values
        # calculate similarity
        sims = wmd_sim('cn',docs1,docs2)
        # calculate SER
        sers = compute_ser(sims)
        # save result as .xls
        save_path = '../../wechat_semantic_similarity/res/'+c+'_wmd2.xls'
        res = pd.DataFrame(columns=['id','REF','HYP','semantic_similarity','SER','WER','difference'])
        res.id = data.id
        res.REF = docs1
        res.HYP = docs2
        res.WER = data.WER
        res.semantic_similarity = sims  
        res.SER = sers
        res.difference = res.SER-res.WER
        res.to_excel(save_path,index=0)
        LogInfo(c+' finish')

def main_en():
    LogInfo('Start')
    path1 = '../data/train_data1.txt'
    path2 = '../data/train_data2.txt'
    data1 = codecs.open(path1,'r',encoding='utf-8').read().split('\r\n')[:-1]
    data2 = codecs.open(path2,'r',encoding='utf-8').read().split('\r\n')[:-1]
    # calculate similarity
    sims = wmd_sim('en',data1,data2)
    # calculate SER
    sers = compute_ser(sims)
    # save result as .xls
    save_path = '../res/res_english_.csv'
    res = pd.DataFrame(columns=['REF','HYP','semantic_similarity','SER'])
    res.REF = data1
    res.HYP = data2
    res.semantic_similarity = sims
    res.SER = sers
    res.to_csv(save_path,index=0)
    LogInfo('Save result as: '+save_path)

if __name__ == '__main__':
    example()
#     main_cn()
 
