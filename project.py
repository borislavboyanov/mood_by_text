#A simple neural network that differentiates between angry and calm songs by their text, and then scraps a lyrics website to find a song by a requested artist and mood

"""
insert into angry_songs (track_id, word, num_count) select track_id, word, num_count from lyrics inner join tids on tids.tid=lyrics.track_id inner join tid_tag on tids.id = tid_tag.tid inner join ta
gs on tags.id = tid_tag.tag where tags.tag='angry';
"""

"""
insert into calm_songs (track_id, word, num_count) select track_id, word, num_count from lyrics inner join tids on tids.tid=lyrics.track_id inner join tid_tag on tids.id = tid_tag.tid inner join
tags on tags.id = tid_tag.tag where tags.tag='calm';
"""

"""
create table calm_and_angry(id int(10) unsigned auto_increment primary key, track_id varchar(20), word varchar(100) COLLATE utf8_unicode_ci, num_count int, mood boolean);
"""

"""
insert into angry_songs (track_id, word, num_count) select track_id, word, num_count from lyrics inner join tids on lyrics.track_id = tids.tid inner join tid_tag on tids.id = tid_tag.tid inner join tags on tid_tag.tag = tags.id where tags.tag = 'angry';
"""

import mysql.connector
import random
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
import numpy as np

db = mysql.connector.connect(user='root', password='', host='localhost', database='song_moods')

angry_cursor = db.cursor(buffered=True)
calm_cursor = db.cursor(buffered=True)
insert_cursor = db.cursor(buffered=True)

angry_cursor.execute('select distinct track_id from angry_songs;')
calm_cursor.execute('select distinct track_id from calm_songs;')
angry_tracks = angry_cursor.fetchall()
calm_tracks = calm_cursor.fetchall()
angry_lyrics = []
calm_lyrics = []

for index, angry in enumerate(angry_tracks):
    lyrics = ''
    angry_cursor.execute('select word, num_count from angry_songs where track_id="' + angry[0] + '";')
    track = angry_cursor.fetchall()
    for word in track:
        lyrics += word[0] + ' '
    lyrics = lyrics[:-1:]
    angry_lyrics.append(lyrics)
    print('Track ' + str(index) + ' out of ' + str(len(angry_tracks)))

for index, calm in enumerate(calm_tracks):
    lyrics = ''
    calm_cursor.execute('select word, num_count from calm_songs where track_id="' + calm[0] + '";')
    track = calm_cursor.fetchall()
    for word in track:
        lyrics += word[0] + ' '
    lyrics = lyrics[:-1:]
    calm_lyrics.append(lyrics)
    print('Track ' + str(index) + ' out of ' + str(len(calm_tracks)))

#angry_cursor.execute('select track_id, word, num_count from angry_songs;')
#calm_cursor.execute('select track_id, word, num_count from calm_songs;')

all_lyrics = []
angry_calm = []
angry_counter = 0
calm_counter = 0

while angry_counter < len(angry_lyrics) or calm_counter < len(calm_lyrics):
    if angry_counter < len(angry_lyrics):
        angry_range = random.randrange(1, 7)
        for _ in range(angry_range):
            all_lyrics.append(angry_lyrics[angry_counter])
            angry_calm.append(0)
            angry_counter += 1
            print('Angry: ' + str(angry_counter))
            if angry_counter == len(angry_lyrics):
                break
    if calm_counter < len(calm_lyrics):
        calm_range = random.randrange(1, 5)
        for _ in range(calm_range):
            all_lyrics.append(calm_lyrics[calm_counter])
            angry_calm.append(1)
            calm_counter += 1
            print('Calm: ' + str(calm_counter))
            if calm_counter == len(calm_lyrics):
                break

max_length = 0
for lyric in all_lyrics:
    if len(lyric) > max_length:
        max_length = len(lyric)

insert_cursor.execute('create table `angry_calm` (id INT(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY,lyrics VARCHAR(' + str(max_length) + ') COLLATE utf8_unicode_ci, mood boolean);')
for index, lyric in enumerate(all_lyrics):
    insert_cursor.execute('insert into angry_calm (lyrics, mood) values ("' + lyric + '", ' + str(angry_calm[index]) + ');')
    db.commit()
vectorizer = CountVectorizer(max_features=5000)
x = vectorizer.fit_transform(all_lyrics).toarray()
x_train, x_test, y_train, y_test = train_test_split(x, angry_calm, test_size=0.1, random_state=0)

new_y_train = []
new_y_test = []

for y in y_train:
    if y == 0:
        new_y_train.append((0, 1))
    else:
        new_y_train.append((1, 0))

for y in y_test:
    if y == 0:
        new_y_test.append((0, 1))
    else:
        new_y_test.append((1, 0))

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=128, activation='relu'))
ann.add(tf.keras.layers.Dense(units=128, activation='relu'))
ann.add(tf.keras.layers.Dense(units=2, activation='softmax'))

#x_train = np.array(x_train)
#x_test = np.array(x_test)
#y_train = np.array(y_train)
#y_test = np.array(y_test)
y_train = np.array(new_y_train)
y_test = np.array(new_y_test)

ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#ann.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
ann.fit(x_train, y_train, batch_size=64, epochs=100)

y_pred = ann.predict(x_test)
final_y_pred = []
for y in y_pred:
    if y[0] > y[1]:
        final_y_pred.append((1, 0))
    else:
        final_y_pred.append((0, 1))

#y_pred = (y_pred > 0.5)
final_y_pred = np.array(final_y_pred)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
cm = multilabel_confusion_matrix(y_test, final_y_pred)
print(cm)
accuracy_score(y_test, final_y_pred)

insert_cursor.execute('select mood from angry_calm;')
angry_calm = insert_cursor.fetchall()
y = []
for i in angry_calm:
    y.append(i[0])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
clf = MultinomialNB().fit(x_train, y_train)
y_pred = clf.predict(x_test)
#

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
#####################################################################################################
#####################################################################################################

import requests
from bs4 import BeautifulSoup, SoupStrainer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import ssl
import string

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('wordnet')
nltk.download('stopwords')

insert_cursor.execute('select word from lyrics;')
words = insert_cursor.fetchall()
db_words = []
for w in words:
    db_words.append(w[0])

result = requests.get('https://WEBSITE/search.php?q=joan+jett')
text = BeautifulSoup(result.text, 'lxml')

panels = text.findAll('div', {'class':'panel'})

panel = 'empty'
for p in panels:
    if 'Artist results' in str(p):
        panel = p

if panel == 'empty':
    print('There are no such artists!')

links = BeautifulSoup(str(panel),parse_only=SoupStrainer('a'))
link = links.findAll('a', href=True)[0]
result = requests.get(link['href'])
songs = BeautifulSoup(result.text, 'lxml')
songs = songs.findAll('div', {'class': 'listalbum-item'})
get_songs = []
for song in songs:
    get_songs.append(song.find('a', href=True))

new_lyrics = []
angry_titles = []
for index, song in enumerate(get_songs):
    result = requests.get(song['href'].replace('..', 'https://WEBSITE'))
    main_text = BeautifulSoup(result.text, 'lxml')
    lyric = main_text.find('div', {'class': 'ringtone'})
    next = lyric.findNextSibling()
    while next.name != 'div':
        next = next.findNextSibling()
    text = next.getText()
    tokenized = nltk.word_tokenize(next.getText())

    flag = True
    while flag:
        flag = False
        for t in tokenized:
            if t in stopwords.words('english'):
                tokenized.remove(t)
                flag = True
                break
            if t in string.punctuation:
                tokenized.remove(t)
                flag = True
                break

    text = []
    for token in tokenized:
        for syn in wordnet.synsets(token):
            text.append(syn.lemmas()[0].name())

    stemmer = PorterStemmer()
    stemmed_text = []
    for word in text:
        stemmed_text.append(stemmer.stem(word))
    flag = True
    while flag:
        flag = False
        for word in stemmed_text:
            if word not in db_words:
                stemmed_text.remove(word)
                flag = True
                break
    stemmed_text = list(dict.fromkeys(stemmed_text))
    x_new_example = ''
    for w in stemmed_text:
        x_new_example += w + ' '
    x_new_example = x_new_example[:-1:]
    x_new_example = [x_new_example]
    song_vector = vectorizer.transform(x_new_example).toarray()
    prediction = ann.predict(song_vector)
    if prediction.flatten()[1] > 0.5:
        title = main_text.find('div', {'class': 'ringtone'})
        title = title.findNextSibling()
        title = title.contents[0]
        title.replace('"', '')
        angry_titles.append(title)
    print('Song ' + str(index + 1) + ' from ' + str(len(get_songs)))
