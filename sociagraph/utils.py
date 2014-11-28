import re as regex
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from random import shuffle
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.metrics import classification_report

# Description: Get the total number of words
# Parameter/s: string
# Return:	   string
def count_words(text):
	return len(text.split(' '))

# Description: Transform string to Text
# Parameter/s: string
# Return:	   Text
def transform_to_text(text):
	return nltk.Text(text)

# Description: Remove non-letters
# Parameter/s: string
# Return:	   string
def remove_non_letters(word):
	return regex.sub('[^A-Za-z ]+', ' ', word)

# Description: Remove non-alphanumeric or non-hyphen, 
# Parameter/s: string
# Return:	   string
def remove_non_alphanumeric(word):
	return regex.sub('[^A-Za-z0-9\- ]+', ' ', word)

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
	return " ".join([w for w in text.split(" ") if not w in stopword_set])

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
	text = regex.sub('[^A-Za-z0-9\- ]+', '', text)
	return nltk.word_tokenize(unicode_to_string(text))

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

# Description: Determine part of speech of a word
# Parameter/s: string
# Return:	   string
def get_pos_tag(text):
	text = regex.sub('[^A-Za-z ]+', ' ', text)
	return nltk.pos_tag(text)

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
	return list(nltk.bigrams(unicode_to_string(text).split(" ")))

# Description: Get the frequency distribution
# Parameter/s: tuple
# Return:	   tuple
def get_frequency_distribution(genre_word):
	return nltk.ConditionalFreqDist(genre_word)

# Description: Get the definition/s of the word
# Parameter/s: string
# Return:	   list
def get_word_definitions(word):
	definitions = []
	for synset in wordnet.synsets(word):
		definitions.append(synset.definition())
	return definitions

# Description: Get the synonyms
# Parameter/s: string
# Return:	   list
def get_synonyms(word):
	synonyms = []
	for synset in wordnet.synsets(word):
		synonyms.append(unicode_to_string(synset.name().split('.')[0]))
	return synonyms

# Description: Check if two words has similar synonyms
# Parameter/s: string | string
# Return:	   boolean
def has_similar_synonyms(word1, word2):
	set1 = set(wordnet.synsets(word1))
	set2 = set(wordnet.synsets(word2))

	return not set1.isdisjoint(set2)

# Description: Get the initial classifications of each word
# Parameter/s: list
# Return:	   tuple
def get_initial_classifications(themes, wordlist):
	classified_wordlist = {}

	for theme in themes:
		for word in wordlist:
			if has_similar_synonyms(theme, word):
				classified_wordlist[word] = theme

	return classified_wordlist

# Description: Builds feature set
# Parameter/s: list | list
# Return:	   list
def build_feature_sets(themes, wordlist, classified_wordlist):
	feature_sets = []
	features = {}

	for word in wordlist:
		if classified_wordlist.has_key(word):
			features['contains(%s)' % word] = True
			feature_sets.append((features, themes.index(classified_wordlist[word])))

	return feature_sets

# Description: Randomize position of list items
# Parameter/s: list
# Return:	   list
def shuffle_set(list):
	shuffle(list)

# Description: Read text file
# Parameter/s: list
# Return:	   list
def read_corpus(list):
	corpus_root = '/usr/share/dict'
	wordlists = PlaintextCorpusReader(corpus_root, '.*')
	wordlists.fileids()
	wordlists.words('connectives')

# Description: Create an SVM classifier
# Parameter/s: None
# Return:	   SklearnClassifier(LinerarSVC())
def create_svm_classifier():
	# from sklearn.feature_extraction.text import TfidfTransformer
	# from sklearn.feature_selection import SelectKBest, chi2
	# from sklearn.naive_bayes import MultinomialNB
	# from sklearn.pipeline import Pipeline
	# pipeline = Pipeline([('tfidf', TfidfTransformer()),
	# 	('chi2', SelectKBest(chi2, k='all')),
	# 	('svm', LinearSVC())])
	return SklearnClassifier(LinearSVC())

# Description: Train the classifier
# Parameter/s: classifier | list
# Return:	   classifier
def train_classifier(classifier, training_data):
	classifier.train(training_data)

	return classifier

# Description: Get the classification report
# Parameter/s: list | list | list
# Return:	   classifier
def get_classification_report(test_classification_index, classification_result, themes):
	report = classification_report(test_classification_index, classification_result, labels=list(set(test_classification_index)),target_names=themes)

	return report

# Description: Convert unicode to string
# Parameter/s: unicode string
# Return:	   string
def unicode_to_string(unicode_string):
	return unicode_string.encode('ascii', 'ignore')

# Description:  Breaks the paragraph into sentences through period (.)
# Parameter/s:  string
# Return:	    list []
# Dependencies: unicode_to_string() |
def paragraph_to_sentences(paragraph):
	# Replace all occurences of period(.) with a single period(.)
	new_paragraph = regex.sub('[.]+', '.', paragraph)

	# Removes unicode
	new_paragraph = unicode_to_string(new_paragraph)

	# Breaks the paragraph to sentences
	return new_paragraph.split('.')


# Description:  Check if the word1 is a synonym of word2
# Parameter/s:  string | string
# Return:	    boolean
def is_synonymous(word1, word2):
	word1_synset = wordnet.synsets(word1)
	word2_synset = wordnet.synsets(word2)

	# Check for any similar synonyms
	for synset in word1_synset:
		if synset in word2_synset:
			return True

	return False


# Description:  Check if the sentence is associated to a label
# Parameter/s:  str | str
# Return:	    dict { word: word_count }
# Dependencies: is_synonymous()
def get_label_associated_words(label, sentence):
	# Remove non-alphanumeric, non-hyphen and non-space
	filtered_sentence = regex.sub('[^A-Za-z0-9\- ]+', '', sentence)

	words = filtered_sentence.split(' ')

	associated_words = { }
			
	for word in words:
		if is_synonymous(word.lower(), label):
			if word not in associated_words:
				associated_words[word] = words.count(word)

	return associated_words

# Description:  Breaks the paragraph into sentences through period (.)
# Parameter/s:  str
# Return:	    list [ (label, sentence, { word: word_count } ), ... ]
# Dependencies: get_label_associated_words()
def get_initial_sentence_classification(labels, sentence_list):
	classified_sentences = []

	for label in labels:
		associated_words = {}

		for sentence in sentence_list:
			associated_words = get_label_associated_words(label, sentence)

			if len(associated_words) > 0:
				classified_sentences.append((sentence, label, associated_words))

	return classified_sentences

# Description:  Get the feature set words
# Parameter/s:  list [ (sentence, label), ... ]
# Return:	    list [ word, ... ]
def get_feature_set_words(labeled_paragraph_list):
	feature_set_words = []
	
	for labeled_paragraph in labeled_paragraph_list:
		for word in tokenize(labeled_paragraph[0]):
			feature_set_words.append(word)

	return set(feature_set_words)