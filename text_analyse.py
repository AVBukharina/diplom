import streamlit as st
from dotenv import load_dotenv
import nltk
from PIL import Image
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
import io
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import string
from summa import summarizer

questions_answers_path = "uploads/questions_and_answers.txt"
change_text_path = 'uploads/change_text.txt'
text_without_question_path = 'uploads/text_without_question.txt'

def text_process(text):
    cloud_text = delete_punctuation(text)
    tokens = del_stopwords(cloud_text)
    cloud_text = nltk.Text(tokens)
    return cloud_text

def delete_punctuation(text):
    spec_chars = string.punctuation + '«»\t—…’'
    new_text = "".join([ch for ch in text if ch not in spec_chars])
    return new_text


# Количество символов с пробелами
def count_symbols_and_words(text):
    count_symbols = len(text)
    text = delete_punctuation(text)
    text_tokens = word_tokenize(text)
    len_text_tokens = len(text_tokens)

    return len_text_tokens, count_symbols


# Разделение по продложениям
def tokenized_text(text):
    tokens = sent_tokenize(text)
    return tokens

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


# Сохрание файла
def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

# Файл вопрос + ответ
def questions_and_answers():
    file = open_file(change_text_path)
    tokens = tokenized_text(file)
    with open(questions_answers_path, "w", encoding="utf-8") as qa:
        for i in range(len(tokens) - 1):
            if '?' not in tokens[i] and '?' in tokens[i + 1]:
                qa.write(tokens[i])
                qa.write('\n')
            else:
                qa.write(tokens[i])
                qa.write(' ')


# Создание списка блоков вопрос + ответ
def list_questions_and_answers():
    text = open_file(questions_answers_path)
    list_blocks = text.split('\n')
    list_qa = []
    for i in list_blocks:
        tokenized_qa = sent_tokenize(i)
        list_qa.append(tokenized_qa)
    return list_qa


# Создание списка с вопросами и списка с ответами
def list_questions_and_list_answers():
    list_qa = list_questions_and_answers()

    questions = []
    answers = []

    for i in list_qa:
        question = ''
        answer = ''
        for j in i:
            if '?' in j:
                question += j
            else:
                answer += j
                answer += ' '
        questions.append(question)
        answers.append(answer)
    return questions, answers


# Создание dataframe с вопросами и ответами
def dataFrame_questions_answers():
    questions, answers = list_questions_and_list_answers()

    data = {'Вопросы': questions,
            'Ответы': answers}

    qa_df = pd.DataFrame(data)
    return qa_df


# Получение текста без вопросов
def text_without_question(text):
    tokens = tokenized_text(text)
    tokenized_texts = []
    for i in tokens:
        if '?' not in i:
            tokenized_texts.append(i)
    with open(text_without_question_path, "w", encoding="utf-8") as t:
        for i in tokenized_texts:
            t.write(i)
            t.write(' ')


# Удаление стоп-слов
def del_stopwords(text):
    lower_text = text.lower()
    tokens = word_tokenize(lower_text)
    #russian_stopwords = stopwords.words("russian")
    with open("stopwords-ru.txt", "r", encoding="utf-8") as file:
        russian_stopwords = file.readlines()
        russian_stopwords = [x.strip() for x in russian_stopwords]

    new_tokens = [token.strip() for token in tokens if token not in russian_stopwords]
    return new_tokens

def show_answers(option, df):
    for i in option:
        st.write(i)
        st.write(df[df['Вопросы'] == i]['Ответы'].values[0])
        if len(option) > 1 and i != option[-1]:
            st.write('______')



from sumy.nlp.tokenizers import Tokenizer
tokenizer = Tokenizer("ru")
from sumy.parsers.plaintext import PlaintextParser

def sumTextRank(text, count_sentence):
    from sumy.summarizers.text_rank import TextRankSummarizer
    parser = PlaintextParser.from_string(text, Tokenizer('ru'))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, count_sentence)
    summary_text = ''
    for i in summary:
        summary_text += str(i)

    return summary_text

def sumLexRank(text, count_sentence):
    from sumy.summarizers.lex_rank import LexRankSummarizer
    parser = PlaintextParser.from_string(text, Tokenizer('ru'))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, count_sentence)
    summary_text = ''
    for i in summary:
        summary_text += str(i)

    return summary_text

def sumLsa(text, count_sentence):
    from sumy.summarizers.lsa import LsaSummarizer
    parser = PlaintextParser.from_string(text, Tokenizer('ru'))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, count_sentence)
    summary_text = ''
    for i in summary:
        summary_text += str(i)

    return summary_text

def sumLuhn(text, count_sentence):
    from sumy.summarizers.luhn import LuhnSummarizer
    parser = PlaintextParser.from_string(text, Tokenizer('ru'))
    summarizer = LuhnSummarizer()
    summary = summarizer(parser.document, count_sentence)
    summary_text = ''
    for i in summary:
        summary_text += str(i)
        summary_text += ' '

    return summary_text

