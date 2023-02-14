
run music_genre_classification.py
run speech_recognition.py

emotion_genre={"happy":"disco","fearful":"classical","disgust":"metal","calm":"country","neutral":"hiphop","angry":"rock","sad":"reggae","surprised":"disco"}

emotion=speech_res()
print(emotion)

for i in emotion:
    playmusic(emotion_genre[i])