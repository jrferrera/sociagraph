import re as regex
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.corpus import wordnet

# Description: Remove non-letters
# Parameter/s: string
# Return:	   string
def remove_non_letters(word):
	return regex.sub('[^A-Za-z ]+', ' ', word)

# Description: Remove extra whitespaces and tabs
# Parameter/s: string
# Return:	   string
def remove_extra_whitespaces(word):
	return " ".join(word.split())

# Description: Checks the lexical diversity of the text
# Parameter/s: string
# Return:	   float
def lexical_diversity(text):
	return len(text) / len(set(text))

# Description: Removes the stopwords in the text
# Parameter/s: string
# Return:	   string
def remove_stopwords(text):
	stopword_set = set(stopwords.words('english'))
	return [w for w in text.split(" ") if not w in stopword_set]

# Description: Checks if the word is in the dictionary
# Parameter/s: string
# Return:	   boolean
def in_dictionary(word):
	return word in nltk.corpus.words.words()

# Description: Gets fractions of the text that are not stopwords
# Parameter/s: string
# Return:	   float
def get_non_stopword_fraction(text):
	stopword_set = set(stopwords.words('english'))
	content = [w for w in text.split(" ") if not w in stopword_set]
	return len(content) / len(text.split(" "))

# Description: Tokenize the text
# Parameter/s: string
# Return:	   tuple
def tokenize(text):
	text = regex.sub('[^A-Za-z ]+', ' ', text)
	return nltk.word_tokenize(text)

# Description: Count the frequeny of each word
# Parameter/s: string
# Return:	   dictionary
def get_bag_of_words(text):
	bag_of_words = dict()
	text = regex.sub('[^A-Za-z ]+', ' ', text)
	text = nltk.word_tokenize(text)

	for word in text:
		bag_of_words[word] = text.count(word)

	return bag_of_words

# Description: Determine part of speech of each word
# Parameter/s: string
# Return:	   tuple
def get_pos_tags(text):
	text = regex.sub('[^A-Za-z ]+', ' ', text)
	text = nltk.word_tokenize(text)

	return nltk.pos_tag(text)

# Description: Get the tag value of the Part Of Speech (POS) tag
# Parameter/s: string
# Return:	   string
def get_pos_tag_value(tag):
	pos_tag_value = {'CC': 'coordinating conjunction', 'CD': 'cardinal number', 'DT': 'determiner', 'EX': 'existential', 'FW': 'foreign word', 'IN': 'preposition/subordinating conjunction', 'JJ':	'adjective', 'JJR': 'adjective comparative', 'JJS': 'adjective superlative', 'LS': 'list marker', 'MD': 'modal', 'NN': 'noun singular or mass', 'NNS': 'noun plural', 'NNP': 'proper noun singular', 'NNPS': 'proper noun plural', 'PDT': 'predeterminer', 'POS': 'possessive ending', 'PRP': 'personal pronoun', 'PRP$': 'possessive pronoun', 'RB': 'adverb', 'RBR': 'adverb comparative', 'RBS': 'adverb superlative', 'RP': 'particle', 'TO': 'to', 'UH': 'interjection', 'VB': 'verb base form', 'VBD': 'verb past tense', 'VBG': 'verb gerund/present participle', 'VBN': 'verb past participle', 'VBP': 'verb singular present non-3d', 'VBZ': 'verb 3rd person singular present', 'WDT':	'wh-determiner', 'WP': 'wh-pronoun', 'WP$': 'possessive wh-pronoun', 'WRB':	'wh-abverb'}[tag]

	return (pos_tag_value.title() if pos_tag_value else 'Unknown')

# Description: Add the POS tag value
# Parameter/s: list
# Return:	   list
def get_pos_tag_values(list):
	for index, word in enumerate(list):
		list[index] = list[index] + (get_pos_tag_value(list[index][1]),)
	return list

# Description: Generate bigrams
# Parameter/s: string
# Return:	   list
def get_bigrams(text):
	return list(nltk.bigrams(text.split(" ")))

# Description: Get the frequency distribution
# Parameter/s: tuple
# Return:	   tuple
def get_frequency_distribution(genre_word):
	return nltk.ConditionalFreqDist(genre_word)

# Description: Get the synonyms
# Parameter/s: word
# Return:	   list
def get_synonyms(word):
	synonyms = []
	for synset in wn.synsets(word):
		synonyms.append(synset)
	return synonyms

# Description: Read text file
# Parameter/s: list
# Return:	   list
def read_corpus(list):
	corpus_root = '/usr/share/dict'
	wordlists = PlaintextCorpusReader(corpus_root, '.*')
	wordlists.fileids()
	wordlists.words('connectives')