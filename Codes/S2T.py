import speech_recognition as sr
r=sr.Recognizer()
with sr.Microphone() as source:
    print("Say:")
    audio=r.listen(source,10)
    print("time over")

try:
    result=r.recognize_google(audio);
    print("Text:"+result);
    
except:
    pass;
