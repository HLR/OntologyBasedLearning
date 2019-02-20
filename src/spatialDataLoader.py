import xml.etree.ElementTree as ET 

# numpy
import numpy as np

# pandas
import pandas as pd

#sklearn
from sklearn.feature_extraction.text import CountVectorizer

def parseSprlXML(sprlXmlfile): 
  
    # parse the xml tree object 
    sprlXMLTree = ET.parse(sprlXmlfile) 
  
    # get root of the xml tree 
    sprlXMLRoot = sprlXMLTree.getroot() 
  
    sentences_list =[]

    # iterate news items 
    for sentenceItem in sprlXMLRoot.findall('./SCENE/SENTENCE'): 
       
        sentence_dic = {}

        sentence_dic["id"] = sentenceItem.attrib["id"]
        
        # iterate child elements of item 
        for child in sentenceItem: 
            
            if child.tag == 'TEXT': 
                sentence_dic[child.tag] = child.text
            elif child.tag == 'LANDMARK' or child.tag == 'TRAJECTOR':
                if "text" in child.attrib:
                    sentence_dic[child.tag] = child.attrib["text"]
                    if "start" in child.attrib :
                        padded_str = ' ' * int(child.attrib["start"]) + child.attrib["text"]
                        sentence_dic[child.tag + "padded"] = padded_str

        sentences_list.append(sentence_dic)

    # create empty dataform for sentences
    sentences_df = pd.DataFrame(sentences_list)
  
    # return sentence items list 
    print("Found number of trajectors and landmarks ", len(sentences_df))

    return sentences_df

output = {'Landmar' : [1, 0], 'Trajector' : [0, 1]}

def getCorpus(sentences_df):

    # Combine landmarks and trajectors phrases and add answers
    corpus_landmarks = pd.DataFrame(sentences_df['LANDMARK']).rename(index=str, columns={"LANDMARK": "Phrase"})
    corpus_landmarks = corpus_landmarks[~corpus_landmarks['Phrase'].isnull()]
    corpus_landmarks['output'] = corpus_landmarks['Phrase'].apply(lambda x: output['Landmar'])

    corpus_trajectors = pd.DataFrame(sentences_df['TRAJECTOR']).rename(index=str, columns={"TRAJECTOR": "Phrase"})
    corpus_trajectors = corpus_trajectors[~corpus_trajectors['Phrase'].isnull()]
    corpus_trajectors['output'] = corpus_trajectors["Phrase"].apply(lambda x: output['Trajector'])

    corpus = corpus_landmarks.append(corpus_trajectors)
    
    print('corpus', type(corpus), len(corpus), corpus.columns)

    # Find distinct words in combined list of phrases
    vectorizer = CountVectorizer()
    vectorizer.fit(corpus["Phrase"].values.astype('U'))

    #print('vectorizer.get_feature_names() ', vectorizer.get_feature_names())

    # Add feature vector for each phrase
    corpus["Phrase_asList"] = corpus['Phrase'].apply(lambda x: [x])
    corpus["Feature_Words"] = corpus['Phrase_asList'].apply(lambda x: vectorizer.transform(x))

    return corpus

#parseSprlXML('data/newSprl2017_all.xml')
#print(getCorpus(newSprl_sentences_df).head(60))
