import streamlit as st
from PIL import Image
import io
from nltk.probability import FreqDist
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from summa import summarizer
import text_analyse as tx
import seaborn as sns

import plotly.express as px

upload_path = "uploads/"
download_path = "downloads/"
transcript_path = "transcripts/"
questions_answers_path = "uploads/questions_and_answers.txt"
change_text_path = 'uploads/change_text.txt'
text_without_question_path = 'uploads/text_without_question.txt'
st.set_page_config(
    page_title="Перевод аудио",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown("# Анализ текста")

text = st.file_uploader("Choose file", type='txt')

state = st.session_state
state.txt = ''

if 'txt' not in state and (text is not None):
    state.txt = text

if text:
    txt = st.text_area('', value=text.read().decode('utf-8'), height=250)

    st.download_button(
        label="Скачать текст",
        data=txt,
        file_name='text.txt',
        mime='text'
    )

    state.txt = txt
    tx.save_file(change_text_path, state.txt)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ['Подсчет слов и символов', 'Вопрос + ответ', 'Облако слов', 'Анализ тональности', 'Суммаризация', 'Ключевые слова'])

with tab1:
    st.header('Количество символов и количество слов')

    col1, col2 = st.columns(2)
    words, count_symbols = tx.count_symbols_and_words(state.txt)

    with col1:
        st.subheader('Количество слов')
        st.text(words)

    with col2:
        st.subheader('Количество символов')
        st.text(count_symbols)

    new_text = tx.text_process(state.txt)

    st.subheader('10 часто встречающихся слов')

    fdist = FreqDist(new_text).most_common(10)
    st.text(fdist)

    fdist = pd.Series(dict(fdist))
    fdist = fdist.to_frame()

    # ## Setting figure, ax into variables
    # fig, ax = plt.subplots(figsize=(5, 5))
    #
    # ## Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
    # all_plot = sns.barplot(x=fdist.index, y=fdist.values, ax=ax)
    # plt.xticks(rotation=30)
    st.bar_chart(fdist)
    # st.pyplot(fig)

with tab2:
    tx.questions_and_answers()
    st.header('Вопрос + ответ')

    list_qa = tx.list_questions_and_answers()
    # for i in list_qa:
    #     for j in i:
    #         st.write(j)
    #     st.write('_______')

    df = tx.dataFrame_questions_answers()
    st.dataframe(df)

    options = st.multiselect('Выберите вопрос', options=df['Вопросы'])

    if st.button('Показать ответы'):
        for i in options:
            st.write(i)
            st.write(df[df['Вопросы'] == i]['Ответы'].values[0])
            if len(options) > 1 and i != options[-1]:
                st.write('______')

with tab3:
    st.header('Облако слов по вашему изображению')
    # Облако всего текста

    cloud_text = tx.text_process(state.txt)

    file = st.file_uploader('Загрузите изображение в формате jpg', type='jpg')
    if file:
        image = file.getvalue()
        mask = np.array(Image.open(io.BytesIO(image)))

    col1, col2 = st.columns(2)
    with col1:
        button_picture = st.button('Сгенерировать облако по картинке')
    with col2:
        button_cloud = st.button('Сгенерировать облако')

    if button_picture and (file is not None):
        text_raw = " ".join(cloud_text)
        wordcloud = WordCloud(width=1600, height=800, mask=mask, background_color='white').generate(text_raw)
        cloud_word = plt.figure(figsize=(20, 10), facecolor='k')
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()

        st.pyplot(cloud_word)


    if button_cloud:
        text_raw = " ".join(cloud_text)
        wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(text_raw)
        cloud_word = plt.figure(figsize=(20, 10), facecolor='k')
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()

        st.pyplot(cloud_word)

with tab4:
    import torch
    import pickle
    from transformers import TextClassificationPipeline, BertTokenizer
    import numpy as np

    model_name = 'models/my_model.sav'
    model = pickle.load(open(model_name, 'rb'))
    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

    labels = ['neutral', 'positive', 'negative']
    def predict_emotion_value(comment):
        piper = pipe(comment)
        list = [d['score'] for d in piper[0]]
        emotion = labels[np.argmax(list)]
        return emotion

    j = 0
    for i in df['Ответы']:
        df['Нейтральные'][j] = pipe(i)[0][0]['score']
        df['Отрицательные'][j] = pipe(i)[0][2]['score']
        df['Положительные'][j] = pipe(i)[0][1]['score']
        j += 1


    df['Нейтральные'] = df['Нейтральные'].transform(lambda x: '{:,.2%}'.format(x))
    df['Отрицательные'] = df['Отрицательные'].transform(lambda x: '{:,.2%}'.format(x))
    df['Положительные'] = df['Положительные'].transform(lambda x: '{:,.2%}'.format(x))
    st.dataframe(df)
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    options = st.multiselect('Выберите вопрос', options=df['Вопросы'], key='sentiment')

    if st.button('Показать результаты'):
        for i in options:
            st.write(i)
            st.write(df[df['Вопросы'] == i]['Ответы'].values[0])
            st.markdown('Нейтральная тональность ' + '<p class="big-font">' +
                        str(df[df['Вопросы'] == i]['Нейтральные'].values[0]) + '</p>' +
                            'Отрицательная тональность ' + '<p class="big-font">' +
                        str(df[df['Вопросы'] == i]['Отрицательные'].values[0]) + '</p>' +
                            'Положительная тональность ' + '<p class="big-font">' +
                        str(df[df['Вопросы'] == i]['Положительные'].values[0]) + '</p>',
                            unsafe_allow_html=True)
            if len(options) > 1 and i != options[-1]:
                st.write('______')

    # for index, row in df.iterrows():
    #     st.write(row['Вопросы'])
    #     st.write(row['Ответы'])
    #     st.markdown('Нейтральная тональность ' + '<p class="big-font">' + str(row['Нейтральные']) + '</p>' +
    #                 'Отрицательная тональность ' + '<p class="big-font">' + str(row['Отрицательные']) + '</p>' +
    #                 'Положительная тональность ' + '<p class="big-font">' + str(row['Положительные'] + '</p>'),
    #                 unsafe_allow_html=True)
    #     st.write('___')


with tab5:
    st.subheader('Определение главных предложений')

    st.markdown("""
        <style>
        .big-font {
            font-size:30px !important;
        }
        </style>
        """, unsafe_allow_html=True)

    count_sentence = st.text_input('Введите колличество предложений после обобщения ответа')
    st.write('Выберите метод для суммаризации')
    choose_textRank = st.checkbox('TextRank')
    choose_lexRank = st.checkbox('LexRank')
    choose_lsa = st.checkbox('LSA')
    choose_luhn = st.checkbox('Luhn')
    options = st.multiselect('Выберите вопрос', options=df['Вопросы'], key='summy')

    if st.button('Суммировать'):
        for i in options:
            st.subheader(i)
            answer = df[df['Вопросы'] == i]['Ответы'].values[0]
            st.write(answer)
            if choose_textRank:
                st.markdown('<p class="big-font">' + 'TextRank' + '</p>' +
                            str(tx.sumTextRank(answer, count_sentence)), unsafe_allow_html=True)
            if choose_lsa:
                st.markdown('<p class="big-font">' + 'Lsa' +
                        '</p>' + str(tx.sumLsa(answer, count_sentence)), unsafe_allow_html=True)
            if choose_luhn:
                st.markdown('<p class="big-font">' + 'Luhn' +
                        '</p>' + str(tx.sumLuhn(answer, count_sentence)), unsafe_allow_html=True)
            if choose_lexRank:
                st.markdown('<p class="big-font">' + 'LexRank' +
                        '</p>' + str(tx.sumLexRank(answer, count_sentence)), unsafe_allow_html=True)
            if len(options) > 1 and i != options[-1]:
                st.write('______')



with tab6:
    st.subheader('Поиск предложений по ключевым словам')
    import re
    from yargy import Parser, rule, or_, and_, not_
    from yargy.predicates import gram
    from yargy.pipelines import morph_pipeline
    from yargy.interpretation import fact
    from yargy.tokenizer import MorphTokenizer
    from razdel import sentenize, tokenize
    from ipymarkup import show_span_ascii_markup as show_markup

    text = state.txt

    keywords_string = st.text_input(label='Введите ключевые слова')

    
    keywords = [token.text for token in tokenize(keywords_string)]


    filtered_keywords = [word for word in keywords if re.fullmatch(r'\w+', word)]

    
    Keyword = fact('Keyword', ['word'])
    KEYWORD_RULE = rule(morph_pipeline(filtered_keywords).interpretation(Keyword.word))

    
    parser = Parser(KEYWORD_RULE)


    sentences = [_.text for _ in sentenize(text)]

    for sentence in sentences:
        matches = []
        spans = []

        for match_ in parser.findall(sentence):
            matches.append(match_)
            spans.extend(_.span for _ in match_.tokens)

       
        if matches:
            st.markdown(f"Предложение: {sentence}")
            show_markup(sentence, spans)
            st.markdown("\n")

