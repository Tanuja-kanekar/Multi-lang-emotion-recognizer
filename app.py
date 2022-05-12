import streamlit as st
import pickle
import librosa
import numpy as np
import pandas as pd
import os

def save_audio(file):
    if file.size > 40000000000:
        return 1
    # if not os.path.exists("audio"):
    #     os.makedirs("audio")
    folder = "audio"
    # clear the folder to avoid storage overload
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0


def main():
    st.sidebar.subheader("Choose a Language of your preference")
    lang_sel = st.sidebar.radio("Audio/Speech",("English","Germany","Greek","Italian"))
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if lang_sel == "English":
        model = pickle.load(open('final1.pkl','rb'))
        st.title("Hello!! I am Speech Emotion Recognizer")
        st.markdown("#### Send the file and I will try my best to predict the emotion")
        audio_file = st.file_uploader("I accept only wav file", type=['wav'])
        if audio_file is not None:
            if not os.path.exists("audio"):
                os.makedirs("audio")
            path = os.path.join("audio", audio_file.name)
            if_save_audio = save_audio(audio_file)
            if if_save_audio == 1:
                st.warning("file size is too large try another file")
            elif if_save_audio == 0:
                st.audio(audio_file,format='audio/wav')
                X, sample_rate = librosa.load(path)
                mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
                feature=mfccs.reshape(1,-1)
                pred = model.predict(feature)
                if pred == ['fear']:
                    st.write("I predicted the emotion as FEAR")
                elif pred == ['sad']:
                    st.write("I predicted the emotion as SAD")
                elif pred == ['happy']:
                    st.write("I predicted the emotion as HAPPY")
                elif pred == ['angry']:
                    st.write("I predicted the emotion as ANGRY")
                else:
                    st.write("I predicted the emotion as DISGUST")
            else:
                st.error("Unknown error")
    elif lang_sel == "Germany":
        model = pickle.load(open('germany.pkl','rb'))
        st.title("Speech Emotion Recognizer(Germany)")
        st.markdown("#### Send the file and I will try my best to predict the emotion")
        audio_file = st.file_uploader("I accept only wav file", type=['wav'])
        if audio_file is not None:
            if not os.path.exists("audio1"):
                os.makedirs("audio1")
            path = os.path.join("audio1", audio_file.name)
            if_save_audio = save_audio(audio_file)
            if if_save_audio == 1:
                st.warning("file size is too large try anoother file")
            elif if_save_audio == 0:
                st.audio(audio_file,format='audio/wav')
                X, sample_rate = librosa.load(path)
                mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
                feature=mfccs.reshape(1,-1)
                pred = model.predict(feature)
                if pred == ['fear']:
                    st.write("I predicted the emotion as FEAR")
                elif pred == ['sad']:
                    st.write("I predicted the emotion as SAD")
                elif pred == ['happy']:
                    st.write("I predicted the emotion as HAPPY")
                elif pred == ['angry']:
                    st.write("I predicted the emotion as ANGRY")
                elif pred == ['boredom']:
                    st.write("I predicted the emotion as BOREDOM")
                elif pred == ['neutral']:
                    st.write("I predicted the emotion as NEUTRAL")
                else:
                    st.write("I predicted the emotion as DISGUST")
            else:
                st.error("Unknown error")
    elif lang_sel == 'Greek':
        model = pickle.load(open('greek.pkl','rb'))
        st.title("Speech Emotion Recognizer(Greek)")
        st.markdown("#### Send the file and I will try my best to predict the emotion")
        audio_file = st.file_uploader("I accept only wav file", type=['wav'])
        if audio_file is not None:
            if not os.path.exists("audio"):
                os.makedirs("audio")
            path = os.path.join("audio", audio_file.name)
            if_save_audio = save_audio(audio_file)
            if if_save_audio == 1:
                st.warning("file size is too large try anoother file")
            elif if_save_audio == 0:
                st.audio(audio_file,format='audio/wav')
                X, sample_rate = librosa.load(path)
                mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
                feature=mfccs.reshape(1,-1)
                pred = model.predict(feature)
                if pred == ['fear']:
                    st.write("I predicted the emotion as FEAR")
                elif pred == ['sad']:
                    st.write("I predicted the emotion as SAD")
                elif pred == ['happy']:
                    st.write("I predicted the emotion as HAPPY")
                elif pred == ['angry']:
                    st.write("I predicted the emotion as ANGRY")
                else:
                    st.write("I predicted the emotion as DISGUST")
            else:
                st.error("Unknown error")
    elif lang_sel == "Italian":
        model = pickle.load(open('italian.pkl','rb'))
        st.title("I am Speech Emotion Recognizer(Italian)")
        st.markdown("#### Send the file and I will try my best to predict the emotion")
        audio_file = st.file_uploader("I accept only wav file", type=['wav'])
        if audio_file is not None:
            if not os.path.exists("audio"):
                os.makedirs("audio")
            path = os.path.join("audio", audio_file.name)
            if_save_audio = save_audio(audio_file)
            if if_save_audio == 1:
                st.warning("file size is too large try anoother file")
            elif if_save_audio == 0:
                st.audio(audio_file,format='audio/wav')
                X, sample_rate = librosa.load(path)
                mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=40).T,axis=0)
                feature=mfccs.reshape(1,-1)
                pred = model.predict(feature)
                if pred == ['fear']:
                    st.write("I predicted the emotion as FEAR")
                elif pred == ['sad']:
                    st.write("I predicted the emotion as SAD")
                elif pred == ['happy']:
                    st.write("I predicted the emotion as HAPPY")
                elif pred == ['angry']:
                    st.write("I predicted the emotion as ANGRY")
                else:
                    st.write("I predicted the emotion as DISGUST")
            else:
                st.error("Unknown error")
    else:
        st.write("Unknown error")


if __name__ == '__main__':
    main()