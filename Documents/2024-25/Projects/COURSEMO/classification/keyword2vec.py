import pdfplumber as pp
import nltk
import re
import os
import spacy
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import keybert
from pdf_scraping import save_csv
import ast

nlp = spacy.load('en_core_web_sm')

def clean_text(text: str):
    text = text.replace('\n', ' ')
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', string=text)
    words = [word for word in text.split(' ') if word != '']
    words_spacy = nlp(' '.join(words))
    words = [word.lemma_ for word in words_spacy]
    valid_words = set(nltk.corpus.words.words())
    stopwords = nltk.corpus.stopwords.words('english')
    words = [word for word in words if word not in stopwords]
    words = [word for word in words if not (len(word) == 1 and word != 'A')]
    words = [word for word in words if word in valid_words]
    newtext = ' '.join(words)
    return newtext

## Keywords

def BERTKeywords(text, n):
    model = keybert.KeyBERT(model='all-mpnet-base-v2')
    keywords = model.extract_keywords(text, top_n=n)
    return keywords

def extract_full_text(pdfstring):
    pdf = pp.open(pdfstring)
    text = ''
    for page in pdf.pages:
        page_text = page.extract_text()
        text = text + ' ' + page_text
    text = clean_text(text)
    return text

def extract_pages(pdfstring, page_num):
    pdf = pp.open(pdfstring)
    text = ''
    for i in range(page_num):
        page_text = pdf.pages[i].extract_text()
        text = text + ' ' + page_text
    text = clean_text(text)
    return text

def get_words(paths):
    texts = [extract_full_text(path) for path in paths]
    words = list(set(' '.join(texts).split()))
    return words

def get_keyword_dicts(paths, all_words):
    keyword_dicts = []
    for path in paths:
        text = extract_full_text(path)
        keywords = BERTKeywords(text, 1000)
        weights = [keywords[i][1] for i in range(len(keywords))]
        scaled_weights = relu(weights, 0)
        keyword_dict = {keywords[i][0]: scaled_weights[i] for i in range(len(keywords))}
        for word in all_words:
            if word in keyword_dict.keys():
                pass
            else:
                keyword_dict[word] = 0
        keyword_dicts.append(keyword_dict)
    print(keyword_dicts)
    return keyword_dicts, keyword_dicts[0].keys()

def sigmoid(array):
    return 1 / (1 + np.exp(-array))

def minmax(weights):
    if max(weights) - min(weights) != 0:
        return [(weight - min(weights)) / (max(weights) - min(weights)) for weight in weights]
    else:
        return weights

def relu(weights, threshold):
    return [max(weight, threshold) for weight in weights]

def get_difference_dicts(dicts):
    difference_dicts = []
    scaled_difference_dicts = []
    for dict in dicts:
        difference_dict = dict.copy()
        for word in dict.keys():
            for dict2 in dicts:
               try:
                   difference_dict[word] = difference_dict[word] - dict2[word] / len(dicts)
               except KeyError:
                   difference_dict[word] = 0
        difference_dicts.append(difference_dict)
    for dict in difference_dicts:
        keywords = list(dict.keys())
        weights = list(dict.values())
        scaled_weights = minmax(relu(weights, 0))
        scaled_difference_dict = {keywords[i]: scaled_weights[i] for i in range(len(keywords))}
        scaled_difference_dicts.append(scaled_difference_dict)
    return difference_dicts, scaled_difference_dicts

def Keyword2Vec(dir, savedir):
    files = os.listdir(dir)
    paths = [dir + '/' + file for file in files]
    labels = [str(file).removesuffix('.pdf') for file in files]
    words = get_words(paths)
    keyword_dicts, keys = get_keyword_dicts(paths, words)
    unscaled_dicts, difference_dicts = get_difference_dicts(keyword_dicts)
    keyword = pd.DataFrame(keyword_dicts).transpose()
    keyword.to_csv('keywords_test.csv')
    unscaled = pd.DataFrame(unscaled_dicts).transpose()
    unscaled.to_csv('unscaled_test.csv')
    model = pd.DataFrame(difference_dicts).transpose()
    print(model)
    model['words'] = keys
    model.rename({str(i): labels[i] for i in range(len(files))}, axis=1)
    model.set_index('words', drop=True, inplace=True)
    model.to_csv(savedir)

def pred(cleantext, model):
    if type(cleantext) == str:
        x = BERTKeywords(cleantext, n=20)
        x_keywords = [x[i][0] for i in range(len(x))]
        x_weights = minmax([x[i][1] for i in range(len(x))])
        model_params = pd.read_csv(model)
        model_params.set_index('words', inplace=True)
        scores = []
        words = list(model_params.index)
        for i in model_params.columns:
            param = []
            for j in range(len(x_keywords)):
                if x_keywords[j] in words:
                    param.append(model_params.loc[str(x_keywords[j]), str(i)])
                else:
                    param.append(0)
            score = np.dot(x_weights, np.array(param))
            scores.append(score)
        if sum(scores) != 0:
            scores = scores / sum(scores)
        return scores
    elif type(cleantext) == list:
        x_full = BERTKeywords(cleantext, n=20)
        model_params = pd.read_csv(model)
        model_params.set_index('words', inplace=True)
        full_scores = []
        words = list(model_params.index)
        for x in x_full:
            x_keywords = [x[i][0] for i in range(len(x))]
            x_weights = minmax([x[i][1] for i in range(len(x))])
            scores = []
            for i in model_params.columns:
                param = []
                for j in range(len(x_keywords)):
                    if x_keywords[j] in words:
                        param.append(model_params.loc[str(x_keywords[j]), str(i)])
                    else:
                        param.append(0)
                score = np.dot(x_weights, np.array(param))
                scores.append(score)
            if sum(scores) != 0:
                scores = scores / sum(scores)
            full_scores.append(scores)
        return full_scores

def model_pred(model, input): 
    if type(input) == str:
        cleantext = clean_text(input)
        return pred(cleantext, model)
    elif type(input) == list:
        cleantext = [clean_text(input[i]) for i in range(len(input))]
        return pred(cleantext, model)

def pred_from_keywords(model, keywords, labels):
    x = keywords
    x_keywords = [x[i][0] for i in range(len(x))]
    x_weights = minmax([x[i][1] for i in range(len(x))])
    model_params = pd.read_csv(model)
    model_params.set_index('words', inplace=True)
    scores = []
    words = list(model_params.index)
    for i in model_params.columns:
        param = []
        for j in range(len(x_keywords)):
            if x_keywords[j] in words:
                param.append(model_params.loc[str(x_keywords[j]), str(i)])
            else:
                param.append(0)
        score = np.dot(x_weights, np.array(param))
        scores.append(score)
    if sum(scores) != 0:
        scores = scores / sum(scores)
    return scores_to_labels(scores, labels)
    
def scores_to_labels(scores: list, labels: list):
    if type(scores) == list:
        max_score = scores.index(max(scores))
    else:
        max_score = scores.tolist().index(max(scores))
    return labels[max_score]

## Frequency

def get_frequency_vector(dir, output):
    files = os.listdir(dir)
    paths = [dir + '/' + file for file in files]
    words = get_words(paths)
    texts = [extract_full_text(path) for path in paths]
    freq_arrays = []
    word_freq_arrays = []
    idf_arrays = []
    for text in texts:
        splittext = text.split()
        word_freq_array = np.array([splittext.count(word) / len(splittext) for word in words])
        freq_array = np.array([splittext.count(word) for word in words])
        freq_arrays.append(freq_array)
        word_freq_arrays.append(word_freq_array)
    avg_array = sum(freq_arrays) / len(freq_arrays)
    for freq_array in freq_arrays:
        idf_array = freq_array / avg_array
        idf_arrays.append(idf_array)
    df = pd.DataFrame([])
    df['words'] = words
    for i in range(len(freq_arrays)):
        df[i] = idf_arrays[i]
    df.set_index('words', inplace=True)
    df.to_csv(output)

def get_composite_model(freqs, model, savedir):
    model_df = pd.read_csv(model)
    freq_df = pd.read_csv(freqs)
    model_df.set_index('words', inplace=True)
    freq_df.set_index('words', inplace=True)
    freq_df = freq_df.reindex(index=model_df.index)
    composite_model = freq_df.to_numpy() * model_df.to_numpy()
    composite_model_df = pd.DataFrame(composite_model)
    composite_model_df['words'] = model_df.index
    composite_model_df.set_index('words', inplace=True)
    composite_model_df.to_csv(savedir)

def get_preds_from_csv(file, output, model, labels):
    questions = pd.read_csv(file)
    question_text = questions['text'].to_list()
    results = model_pred(model, question_text)
    df = pd.DataFrame(results)
    df['pred1'] = [scores_to_labels(results[i], labels) for i in range(len(results))]
    df['correct_label1'] = questions['label1'].to_list()
    df['correct_label2'] = questions['label2'].to_list()
    df['text'] = question_text
    df.to_csv(output)

def get_accuracy(file):
    df = pd.read_csv(file)
    preds = df['pred1'].to_numpy().tolist()
    true = df['correct_label'].to_numpy().tolist()
    correct_preds = [1 if preds[i] == true[i] else 0 for i in range(len(preds))]
    return sum(correct_preds) / len(correct_preds)

def ensemble(subject, name):
    Keyword2Vec('data/{}/{}'.format(subject, name), '{}-{}.csv'.format(subject, name))
    get_frequency_vector('data/{}/{}'.format(subject, name), '{}-{}-freq.csv'.format(subject, name))
    get_composite_model('{}-{}-freq.csv'.format(subject, name), '{}-{}.csv'.format(subject, name), '{}-{}.csv'.format(subject, name))
    os.remove('{}-{}-freq.csv'.format(subject, name))

