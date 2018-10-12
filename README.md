# Text similarity: WMD_enhanced
Compute text similarity using Word Mover's Distance algorithm (Enhanced) 

## Dependencies    
python 3.6.5   
pyemd, numpy, gensim, sklearn, nltk, jieba, pandas, codecs, re

## Pretrained word2vec model used in this code
Support both English and Chinese text format   
Chinese word2vec CBOW: utf8  2.18G   
http://pan.baidu.com/s/1qX334vE     
English word2vec 1.5G     
https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download

## Algorithm description
Enhance the Word Mover's Distance algorithm.
### What is Word Mover's Distance algorithm?
See algorithm details in this paper: 
"From Word Embeddings To Document Distances": http://proceedings.mlr.press/v37/kusnerb15.pdf
### What are the shortcomings of WMD algorithm?
1.WMD algorithm is time consuming due to the computation of Euclidean distance between word vectors. 
2.WMD algorithm would compute the distance between two documents, which is difficult to compute similarity in [0,1]   
### What are the enhancements?
WMD_enhanced algorithm compute the Normalized Cosine Distance between word vectors. 
Steps:
1. Compute cosine similarity between word vectors as res1
2. Normalize the value of res1 to [0,1] as res2
3. Normalized Cosine Distance = 1-res2

## Example usage

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

## Example results
### English:   
man sitting using tool at a table in his home.
The president comes to China
Similarity: 0.1213
SER: 0.8787

vegetable is being sliced.
someone is slicing a tomato with a knife on a cutting board.
Similarity: 0.3555
SER: 0.6445  

a speaker presents some products
the speaker is introducing the new products on a fair.
Similarity: 0.4823
SER: 0.5177     

### Chinese:      
时间太晚不得就算了
此间贷款不得就算啦
Similarity: 0.0903
SER: 0.9097       

他整天愁眉苦脸
他和朋友去逛街
Similarity: 0.1572
SER: 0.8428

学无止境
学海无涯，天道酬勤
Similarity: 0.4049
SER: 0.5951        


