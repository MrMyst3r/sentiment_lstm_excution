import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pyaudio
import whisper
import time
import threading
from googletrans import Translator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from janome.tokenizer import Tokenizer as JanomeTokenizer

# 감정 전이 확률 행렬
P = np.array([[0.195039, 0.004510, 0.101466, 0.162345, 0.201804, 0.181511, 0.153326],
              [0.025850, 0.145045, 0.356869, 0.128052, 0.027525, 0.227382, 0.089277],
              [0.033709, 0.057450, 0.301810, 0.184811, 0.031348, 0.298269, 0.092602],
              [0.038859, 0.009447, 0.122103, 0.425312, 0.044385, 0.237255, 0.122638],
              [0.057446, 0.002611, 0.018816, 0.058137, 0.630366, 0.055218, 0.177406],
              [0.041398, 0.013869, 0.157397, 0.194873, 0.046301, 0.439619, 0.106542],
              [0.043883, 0.006775, 0.064611, 0.159960, 0.173711, 0.131547, 0.419515]])

# 감정 목록
emotions = ['Neutral', 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

# DataFrame 생성
transition_matrix = pd.DataFrame(P, index=emotions, columns=emotions)


# Whisper 모델 로드
stt_model = whisper.load_model("medium")

# 1. 모델 불러오기 (HDF5 형식으로 저장된 모델 로드)
emotion_model = load_model('C:/models/emotion_lstm_model.h5')

# 2. Tokenizer 불러오기
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 3. LabelEncoder 불러오기
with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

# 4. 형태소 분석기 설정 (Janome)
janome_tokenizer = JanomeTokenizer()

# 일본어 문장을 형태소 분석하여 토큰화하는 함수
def tokenize_japanese_sentence(sentence):
    tokens = janome_tokenizer.tokenize(sentence, wakati=True)
    return " ".join(tokens)

def record_audio_for_10_seconds():
    sample_rate = 16000
    chunk_size = 1024
    audio_format = pyaudio.paInt16
    channels = 1
    frames = []

    # PyAudio 객체 생성
    audio = pyaudio.PyAudio()

    # 스트림 열기
    stream = audio.open(format=audio_format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    st.write("녹음 중입니다... (10초)")
    start_time = time.time()

    while time.time() - start_time < 10:  # 5초간 녹음
        data = stream.read(chunk_size, exception_on_overflow=False)
        frames.append(data)

    # 스트림 종료
    stream.stop_stream()
    stream.close()
    audio.terminate()
    st.write("녹음이 종료되었습니다.")
    
    # 오디오 데이터를 float32로 변환 (Whisper는 float32 형식을 요구함)
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0

    # Whisper 모델을 사용하여 텍스트로 변환
    result = stt_model.transcribe(audio_data, language="ja")
    return result['text']

translator = Translator()  # Google 번역기 객체 생성

# 한국어 음성 문장을 일본어로 번역하는 함수 (10초 동안 녹음)
def transcribe_ko_directly():
    sample_rate = 16000
    chunk_size = 1024
    audio_format = pyaudio.paInt16
    channels = 1
    frames = []

    # PyAudio 객체 생성
    audio = pyaudio.PyAudio()

    # 스트림 열기
    stream = audio.open(format=audio_format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    st.write("녹음 중입니다... (10초)")
    start_time = time.time()

    while time.time() - start_time < 10:  # 10초 동안 녹음
        data = stream.read(chunk_size, exception_on_overflow=False)
        frames.append(data)

    # 스트림 종료
    stream.stop_stream()
    stream.close()
    audio.terminate()
    st.write("녹음이 종료되었습니다.")

    # 오디오 데이터를 float32로 변환 (Whisper는 float32 형식을 요구함)
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0

    # Whisper 모델을 사용하여 한국어 텍스트로 변환
    result = stt_model.transcribe(audio_data, language="ko")  # 한국어로 변환
    korean_text = result['text']

    # 한국어 텍스트를 일본어로 번역
    translated = translator.translate(korean_text, src='ko', dest='ja')
    japanese_text = translated.text

    # 두 문장을 하나의 텍스트 영역에 표시
    combined_text = f"녹음된 한국어 텍스트 :\n{korean_text}\n\n번역된 일본어 텍스트 :\n{japanese_text}"
    st.text_area("녹음/번역 텍스트", combined_text, height=200)

    return japanese_text


# Streamlit UI 구성
st.title("감정 예측 시연 - Execution")

# 사이드바에 페이지 선택 추가
tabs = st.tabs(["INPUT : TEXT", "INPUT : JP Voice", "INPUT : KR Voice"])

# 페이지 1
with tabs[0]:
    st.title("INPUT : TEXT")
    st.write("일본어 문장을 입력해보세요!")
    
    # 사용자 입력 받기
    user_input = st.text_area("여기에 일본어 문장을 입력하세요", height=70)
    
    if user_input:
        # 결과 문자열 생성
        results = ""
        
        # 원본 문장
        results += "입력된 문장:\n" + user_input + "\n\n"
        
        # 형태소 분석 결과
        tokenized_sentence = tokenize_japanese_sentence(user_input)
        results += "형태소 분석 결과:\n" + tokenized_sentence + "\n\n"
        
        # 단어 토큰화 결과
        tokenized_new_sentences = [tokenize_japanese_sentence(sentence) for sentence in [user_input]]
        results += "단어 토큰화 결과:\n" + tokenized_new_sentences[0] + "\n\n"
        
        # 벡터화된 결과
        new_sequences = tokenizer.texts_to_sequences(tokenized_new_sentences)
        results += "벡터화된 결과:\n" + str(new_sequences[0]) + "\n\n"
        
        # 패딩된 벡터 결과
        max_len = 100
        new_padded = pad_sequences(new_sequences, maxlen=max_len)
        results += "패딩된 벡터화 결과:\n" + str(new_padded[0]) + "\n\n"
        
        # 감정 예측
        predictions = emotion_model.predict(new_padded)
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_emotion = label_encoder.inverse_transform([predicted_labels[0]])[0]
        results += "예측된 감정:\n" + predicted_emotion
        
        # 모든 결과를 하나의 st.text_area에 표시
        st.text_area("처리 결과", value=results, height=400)
        
        # 감정 전이 확률 시각화
        # 그래프 생성
        fig, ax = plt.subplots()
        ax.bar(emotions, transition_matrix.loc[predicted_emotion].values)
        
        # 그래프 제목과 레이블 설정
        ax.set_title('Emotion Transition Probability')
        ax.set_xlabel('To Emotion')
        ax.set_ylabel('Probability')

        # Streamlit에 그래프 표시
        st.pyplot(fig)
        
# 페이지 2
with tabs[1]:
    st.title("INPUT : JP Voice")
    st.write("일본어로 말해보세요!")

    if st.button("녹음 시작(10sec)", key="jp_voice_record"):
        transcription = record_audio_for_10_seconds()

        # 녹음이 끝난 후 전체 문장에 대한 감정 분석 수행
        new_sentences = [transcription]
        tokenized_new_sentences = [tokenize_japanese_sentence(sentence) for sentence in new_sentences]
        new_sequences = tokenizer.texts_to_sequences(tokenized_new_sentences)
        max_len = 100
        new_padded = pad_sequences(new_sequences, maxlen=max_len)
        predictions = emotion_model.predict(new_padded)
        predicted_labels = np.argmax(predictions, axis=1)

        # 예측된 감정 출력
        predicted_emotion = label_encoder.inverse_transform([predicted_labels[0]])[0]
        st.text_area("녹음된 텍스트 ( 예상 감정 )", f'{transcription} ( {predicted_emotion} )')
    
    
# 페이지 3
with tabs[2]:
    st.title("INPUT : KR Voice")
    st.write("한국어로 말해보세요!")

    if st.button("녹음 시작(10sec)", key="kr_voice_record"):
        transcription = transcribe_ko_directly()

        # 녹음이 끝난 후 전체 문장에 대한 감정 분석 수행
        new_sentences = [transcription]
        tokenized_new_sentences = [tokenize_japanese_sentence(sentence) for sentence in new_sentences]
        new_sequences = tokenizer.texts_to_sequences(tokenized_new_sentences)
        max_len = 100
        new_padded = pad_sequences(new_sequences, maxlen=max_len)
        predictions = emotion_model.predict(new_padded)
        predicted_labels = np.argmax(predictions, axis=1)

        # 예측된 감정 출력
        predicted_emotion = label_encoder.inverse_transform([predicted_labels[0]])[0]
        st.text_area("번역된 텍스트 ( 예상 감정 )", f'{transcription} ( {predicted_emotion} )')
