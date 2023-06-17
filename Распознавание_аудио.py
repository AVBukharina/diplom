import streamlit as st
import streamlit.components.v1 as components
import os
import whisper
import st_pages
import audio as a

upload_path = 'uploads/'
download_path = "downloads/"
transcript_path = "transcripts/"
questions_answers_path = '/questions_and_answers.txt'
change_text_path = '/change_text.txt'
text_without_question_path = '/text_without_question.txt'


def save_transcript(transcript_data, txt_file):
    with open(os.path.join(transcript_path, txt_file), "w", encoding='utf-8') as f:
        f.write(transcript_data)


st.markdown("# Распознавание аудио")

audio = st.file_uploader("Загрузите аудио")

if audio is not None:
    audio_bytes = audio.read()
    with open(os.path.join(download_path, audio.name), "wb") as f:
        f.write(audio.getbuffer())

    with st.spinner(f"Загрузка аудио ... "):
        output_audio_file = audio.name.split('.')[0] + '.mp3'
        if audio.name.split('.')[0] != 'mp3':
            audio = a.to_mp3(audio, output_audio_file, download_path)
        # output_audio_file = to_mp3(audio, output_audio_file, upload_path, download_path)
        audio_file = open(os.path.join(download_path, output_audio_file), 'rb')
        audio_bytes = audio_file.read()
    print("Открытие аудио", audio_file)
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("Можете прослушать ваше аудио")
        st.audio(audio_bytes)
    with col2:
        whisper_model_type = st.radio("Выберите модель для распознавания аудио",
                                      ('Tiny', 'Base', 'Small', 'Medium', 'Large'))

    if st.button("Транскрибировать"):
        with st.spinner(f"Подождите идет перевод"):
            if whisper_model_type == 'Tiny':
                transcript = a.process_audio(str(os.path.abspath(os.path.join(download_path, output_audio_file))),
                                             a.load_models_tiny())
            elif whisper_model_type == 'Base':
                transcript = a.process_audio(str(os.path.abspath(os.path.join(download_path, output_audio_file))),
                                             a.load_models_base())
            elif whisper_model_type == 'Small':
                transcript = a.process_audio(str(os.path.abspath(os.path.join(download_path, output_audio_file))),
                                             a.load_models_small())
            elif whisper_model_type == 'Medium':
                transcript = a.process_audio(str(os.path.abspath(os.path.join(download_path, output_audio_file))),
                                             a.load_models_medium())
            elif whisper_model_type == 'Large':
                transcript = a.process_audio(str(os.path.abspath(os.path.join(download_path, output_audio_file))),
                                             a.load_models_large())

            output_txt_file = str(output_audio_file.split('.')[0] + ".txt")
            save_transcript(transcript, output_txt_file)

            output_file = open(os.path.join(transcript_path, output_txt_file), "r", encoding='utf-8')
            output_file_data = output_file.read()

            st.balloons()
            st.success('Перевод завершен')

            st.download_button(
                    label="Загрузить",
                    data=output_file_data,
                    file_name=output_txt_file,
                    mime='text/plain'
            )
else:
    st.warning('Пожалуйста загрузите Ваше аудио')

# if audio:
#     file_name = audio.name.split('.')[0]
#     print(file_name)
#
#     st.audio(audio)
#     audio_bytes = audio.read()
#
#     with open(os.path.join(download_path, audio.name), 'wb') as f:
#         f.write(audio.getbuffer())

# transcriber = AudioRecognition(OPENAI_API_KEY)
# text = transcriber.transcribe(f'{download_path}{audio.name}')
# st.text(text)
