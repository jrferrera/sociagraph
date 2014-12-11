import re as regex
import operator
from random import shuffle

import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# Reference for nltk: Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O'Reilly Media Inc.

# Description: Get the total number of words
# Parameter/s: string
# Return:	   string
# Dependecies: remove_extra_whitespaces()
def count_words(text):
	return len(remove_extra_whitespaces(text).split(' '))

# Description: Get the total number of vocabulary or unique words
# Parameter/s: string
# Return:	   string
# Dependecies: tokenize() | remove_extra_whitespaces()
def get_vocabulary_count(text):
	return len(set(tokenize(remove_extra_whitespaces(text))))


# Description: Transform string to Text
# Parameter/s: string
# Return:	   Text
def transform_to_text(text):
	return nltk.Text(text)

# Description: Get the stem of the word
# Parameter/s: string
# Return:	   string
def stem(word):
	stemmer = nltk.LancasterStemmer()
	return stemmer.stem(word)


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

# Description: Remove spaces
# Parameter/s: string
# Return:	   string
def remove_spaces(word):
	return regex.sub("[ ]", '', word)

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
	text = regex.sub('[^A-Za-z0-9\.\- ]+', '', text)
	return nltk.word_tokenize(unicode_to_string(text))

# Description: Sort a dictionary by value
# Parameter/s: dictionary
# Return:	   list
# Reference: http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
def sort_dictionary_by_value(dictionary, reverse = False):
	if not reverse:
		return sorted(dictionary.items(), key = operator.itemgetter(1))
	else:
		return sorted(dictionary.items(), key = operator.itemgetter(1)).reverse()

# Description: Sort a dictionary by key
# Parameter/s: dictionary
# Return:	   list
# Reference: http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
def sort_dictionary_by_key(dictionary, reverse = False):
	if not reverse:
		return sorted(dictionary.items(), key = operator.itemgetter(0))
	else:
		return sorted(dictionary.items(), key = operator.itemgetter(0)).reverse()

# Description: Count the frequeny of each word
# Parameter/s: string
# Return:	   dict
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

# Description: Check if possible keyword
# Parameter/s: (..., )
# Return:	   bool
# Dependencies: get_pos_tag()
def is_possible_keyword(word_list):
	tags = ''
	pattern = regex.compile(r"(ADJ ?)* (NN ?)+$|(NN ?)+$|(NN ?)+IN (NN ?)+$")

	for word in word_list:
		tags += nltk.pos_tag(lemmatize(word).split(" "))[0][1] + " "

	tags = tags.rstrip()

	return True if pattern.match(tags) is not None else False

# Description: Generate bigrams
# Parameter/s: str | int
# Return:	   list [ (..., ), ...]
# Dependencies: tokenize()
def get_ngrams(text, word_count):
	return nltk.ngrams(tokenize(text), word_count)

# Description: Generate bigrams
# Parameter/s: str
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
# Return:	   list | None
def get_word_definitions(word):
	definitions = []
	for synset in wordnet.synsets(word):
		definitions.append(unicode_to_string(synset.definition()))

	if len(definitions) == 0:
		definitions = None
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
# Return:	   SklearnClassifier(LinearSVC())
def create_svm_classifier():
	# return SklearnClassifier(SVC(probability = True))
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
# def get_classification_report(test_classification_index, classification_result, themes):
# 	report = classification_report(test_classification_index, classification_result, labels=list(set(test_classification_index)),target_names=themes)

# 	return report

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

	if new_paragraph.endswith('.'):
		new_paragraph = new_paragraph[:-1]
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

# Description:  Assigns theme
# Parameter/s:  list [ item, item ] | str
# Return:	    list [ (word, theme) ]
# Dependencies:	unicode_to_string()
def assign_theme(labeled_corpora, theme):
	labeled_text = []
	
	for corpus in labeled_corpora:
		labeled_text.append((unicode_to_string(corpus['text']), theme))

	return labeled_text

# Description:  Get the features
# Parameter/s:  str | list
# Return:	    dict { contains(word): True, is_synonymous }
# Dependencies: tokenize() | is_synonymous() | lemmatize()
def get_features(text, feature_sets_words, theme):
	features = {}

	for word in tokenize(text.lower()):
		features.update({
			'contains(' + word + ')': lemmatize(word) in feature_sets_words,
			'synonymous_to_theme(' + word + ')': is_synonymous(lemmatize(word), lemmatize(theme))
			})

	return features


# Description:  Get the feature sets
# Parameter/s:  [ (word, theme) ... ] | [ word, ... ] | str
# Return:	    list [({ contains(word): True })]
# Dependencies: get_features()
def get_theme_corpus_feature_sets(combined_labeled_text, feature_set_words, theme):
	feature_sets = [ ( get_features(item[0].lower(), feature_set_words, theme), item[1]) for item in combined_labeled_text ]
	
	return feature_sets

# Description:  Get the most frequent sentiment
# Parameter/s:  dict { str: int, ... }
# Return:	    str
def get_most_frequent_sentiment(sentiment_frequencies):
	max_sentiment = ''
	max_frequency = None

	for sentiment in sentiment_frequencies:
		if max_frequency == None:
			max_sentiment = sentiment
			max_frequency = sentiment_frequencies[sentiment]
		else:
			if sentiment_frequencies[sentiment] > max_frequency:
				max_sentiment = sentiment
				max_frequency = sentiment_frequencies[sentiment]

	return max_sentiment


# Description:  Get the sentiment features
# Parameter/s:  str | list
# Return:	    dict { contains(word): True, synonymous }
# Dependencies: tokenize() | is_synonymous() | lemmatize()
def get_sentiment_features(text, feature_set_words, sentiments):
	features = {}

	for word in tokenize(text.lower()):
		features.update({
			'contains(' + word + ')': lemmatize(word) in feature_set_words
			})

		for sentiment in sentiments:
			features['synonymous_to_' + sentiment + '(' + word + ')'] = is_synonymous(lemmatize(word), sentiment)

	return features


# Description:  Get the feature sets
# Parameter/s:  [ (word, theme) ... ] | [ word, ... ] | str
# Return:	    list [({ contains(word): True })]
# Dependencies: get_features()
def get_sentiment_feature_sets(combined_labeled_text, feature_set_words, sentiments):
	# feature_sets = [ ({ word: (lemmatize(word) in tokenize(item[0])) for word in feature_set_words }, item[1]) for item in combined_labeled_text ]
	feature_sets = [ (get_sentiment_features(item[0], feature_set_words, sentiments), item[1]) for item in combined_labeled_text ]
	# feature_sets = [ ( get_features(item[0].lower(), feature_set_words, theme), item[1]) for item in combined_labeled_text ]
	
	return feature_sets

# Description:  Get the accuracy score
# Parameter/s:  list [ str, ... ] | list [ str, ... ] 
# Return:	    float
def get_accuracy_score(correct_labels, test_labels):
	return accuracy_score(correct_labels, test_labels)

# Description:  Get the precision score
# Parameter/s:  list [ str, ... ] | list [ str, ... ] 
# Return:	    float
def get_precision_score(correct_labels, test_labels):
	return precision_score(correct_labels, test_labels, labels=None)

# Description:  Get the recall score
# Parameter/s:  list [ str, ... ] | list [ str, ... ] 
# Return:	    float
def get_recall_score(correct_labels, test_labels):
	return recall_score(correct_labels, test_labels, labels=None)

# Description:  Get the f1 score
# Parameter/s:  list [ str, ... ] | list [ str, ... ] 
# Return:	    float
def get_f_measure_score(correct_labels, test_labels):
	return f1_score(correct_labels, test_labels, labels=None)

# Description:  Get the accuracy, precision, recall and f-measure
# Parameter/s:  list [ str, ... ] | list [ str, ... ]  | list [ str, ... ]
# Return:	    dict ( { 'str': float } )
def get_classification_scores(correct_labels, test_labels, labels):
	classification_scores = {}

	true_values = [ labels.index(label) for label in correct_labels ]
	predicted_values = [ labels.index(label) for label in test_labels ]

	classification_scores = {
		'accuracy': accuracy_score(correct_labels, test_labels),
		'precision': precision_score(true_values, predicted_values),
		'recall': recall_score(true_values, predicted_values),
		'f-measure': f1_score(true_values, predicted_values),
	}

	return classification_scores

# Description:  Get the accuracy, precision, f1 and recall score
# Parameter/s:  list [ str, ... ] | list [ str, ... ]
# Return:	    str
def get_classification_report(correct_labels, test_labels, theme):
	not_theme = 'not_' + theme
	target_names = [theme, not_theme]
	return classification_report(correct_labels, test_labels, target_names=target_names)

# Description:  Lemmatize the string / Covert to singular sense
# Parameter/s:  str
# Return:	    str
def lemmatize(string):
	lemmatizer = WordNetLemmatizer()

	return lemmatizer.lemmatize(string)

# similarity

# from nltk.corpus import wordnet as wn
# 	similarts = []

# 	Aword = 'language'
# 	Bword = 'barrier'

# 	synsetsA = wn.synsets(Aword)
# 	synsetsB = wn.synsets(Bword)

# 	groupA = [wn.synset(str(synset.name())) for synset in synsetsA]
# 	groupB = [wn.synset(str(synset.name())) for synset in synsetsB]

# 	for sseta in groupA:
# 		for ssetb in groupB:
# 			path_similarity = sseta.path_similarity(ssetb)
# 			wup_similarity = sseta.wup_similarity(ssetb)

# 			if path_similarity is not None:
# 				similars.append({
# 					'path':path_similarity,
# 					'wup':wup_similarity,
# 					'wordA':sseta,
# 					'wordB':ssetb,
# 					'wordA_definition':sseta.definition(),
# 					'wordB_definition':ssetb.definition()
# 				})
# Sorting similarity probability
# similars = sorted(similars, key=lambda item: item['path'], reverse=True)


# Organized printing
# for item in similars:
# 	print item['wordA'],"-",item['wordA_definition']
# 	print item['wordB'],"-",item['wordB_definition']
# 	print 'Path similarity - ', item['path'],'\n'

# def get_features_summary(feature_sets):
# 	features_summary = {}
# 	for features, label in feature_sets:
# 		for 