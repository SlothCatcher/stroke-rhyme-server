"""Russian rhyme inference through RhymeTagger and eSpeak NG."""
import re
import threading
from functools import lru_cache
from typing import Annotated
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rhymetagger import RhymeTagger

app = FastAPI(title='Строка: анализ рифм')
lock = threading.Lock()
model = RhymeTagger()
model.load_model('ru', verbose=False)
IPA_VOWELS = set('aɑɐɒæeɛəɜɞɪiɨɯoɔɵœøuʊʌyʉɤ')
VOWELS = set('аеёиоуыэюя')

class Request(BaseModel):
    terms: Annotated[list[str], Field(min_length=1, max_length=350)]

@lru_cache(maxsize=4096)
def transcribe(term: str):
    # eSpeak does not interpret U+0301. When the vowel counts align, place
    # the IPA stress mark at the explicitly stressed vowel (within its word).
    ipa = model._transcription(term.replace('\u0301', '')).strip()
    if '\u0301' not in term:
        return ipa, False
    words = term.split()
    ipa_words = ipa.split()
    if len(words) != len(ipa_words):
        return ipa, True
    result=[]
    uncertain=False
    for word, spoken in zip(words, ipa_words):
        if '\u0301' not in word:
            result.append(spoken)
            continue
        marker=word.rfind('\u0301')
        stressed=sum(ch.lower() in VOWELS for ch in word[:marker])-1
        spoken=spoken.replace('ˈ','').replace('ˌ','')
        positions=[i for i,ch in enumerate(spoken) if ch in IPA_VOWELS]
        if stressed < 0 or len(positions)!=sum(ch.lower() in VOWELS for ch in word) or stressed>=len(positions):
            result.append(model._transcription(word.replace('\u0301','')).strip())
            uncertain=True
        else:
            i=positions[stressed]
            result.append(spoken[:i]+'ˈ'+spoken[i:])
    return ' '.join(result), uncertain

@app.get('/api/health')
def health():
    return {'ok':True,'model':'RhymeTagger ru + eSpeak NG'}

@app.post('/api/rhymes')
def rhymes(request:Request):
    if any(len(t)>200 or not re.fullmatch(r'[а-яА-ЯёЁ\u0301\s-]+',t) for t in request.terms):
        raise HTTPException(400,'Недопустимый текст для анализа')
    try:
        with lock:
            transcriptions=[transcribe(t) for t in request.terms]
            inputs=[{'text':t,'ipa':ipa} for t,(ipa,_) in zip(request.terms,transcriptions)]
            labels=model.tag(inputs,transcribed=True,output_format=3,window=min(350,len(inputs)),same_words=True,ngram=0,fast_ipa=True)
        if len(labels)!=len(request.terms):
            raise ValueError('unexpected model output length')
        return {'labels':labels,'stress_uncertain':[i for i,(_,warning) in enumerate(transcriptions) if warning], 'model':'RhymeTagger ru + eSpeak NG'}
    except Exception:
        raise HTTPException(503,'Модель не смогла обработать текст')
