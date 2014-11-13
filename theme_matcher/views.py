from django.shortcuts import render
from django.http import HttpResponse

# Import utilities in utils.py
import nltk
from theme_matcher.utils import remove_non_letters
from theme_matcher.utils import remove_extra_whitespaces
from theme_matcher.utils import remove_stopwords
from theme_matcher.utils import tokenize
from theme_matcher.utils import get_bag_of_words
from theme_matcher.utils import get_pos_tags
from theme_matcher.utils import get_pos_tag_values
from theme_matcher.utils import get_bigrams
from theme_matcher.utils import in_dictionary
from theme_matcher.utils import get_synonyms
from theme_matcher.utils import get_word_definitions
from theme_matcher.utils import get_pos_tag
from theme_matcher.utils import get_pos_tag_value
from theme_matcher.utils import get_initial_classifications
from theme_matcher.utils import build_feature_sets
from theme_matcher.utils import has_similar_synonyms
from theme_matcher.utils import shuffle_set
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.metrics import classification_report

def index(request):
	template_name = 'theme_matcher/index.html'

	return render(request, template_name)

def results(request):
	template_name = 'theme_matcher/results.html'

	themes = request.POST.get('theme', False)
	original_text = request.POST.get('text', False)

	# original_text = "Many of the trainings he  attended  in the last  two years  were   on   organic  agriculture, climate  change  and even hydrophonics .  to name  a few are as  follows;    Trainer on Training (TOT )on Organic farming (requested by Albay DTI), Trainer on Training (TOT) on Climate Change, Trainer on Training (TOT) on Organic farming and Organic fertilizers, Training on Vermicomposting, Training on Organic Swine and Poultry, Training on Hydrophonics KSK (Kaisa sa Kasaganaan): vegetables, producing high value commercial crops (December to April)"
	
	# Process the themes
	themes = remove_non_letters(themes)
	themes = remove_extra_whitespaces(themes)
	themes = tokenize(themes)

	# Process the tokens
	tokens = tokenize(original_text)

	# Process the Bag of Words
	bag_of_words = get_bag_of_words(original_text)

	# Process POS Tagging
	pos_tags = get_pos_tags(original_text)
	pos_tags = get_pos_tag_values(pos_tags)

	# Process bigrams
	bigrams = get_bigrams(original_text)

	# Process theme definitions
	theme_definitions = []
	for theme in themes:
		theme_definitions.append([theme, get_word_definitions(theme)])

	filtered_text = remove_stopwords(original_text)
	filtered_text = tokenize(filtered_text.lower())

	# Process text without stopwords, punctuations, numbers
	# Removed words not in English dictionary
	new_filtered_words = []
	for word in filtered_text:
		if in_dictionary(word):
			new_filtered_words.append(word)
	
	# Get the classification of each word based on synonyms
	classified_wordlist = get_initial_classifications(themes, new_filtered_words)

	# Process feature sets
	feature_sets = build_feature_sets(themes, new_filtered_words, classified_wordlist)
	shuffle_set(feature_sets)

	# Use n% of the feature sets as test set
	size = int(len(feature_sets) * 0.20)
	size = 2 if size < 2 else size
	training_set = feature_sets[size:]
	test_set = feature_sets[:size]

	# Build the classifier
	classification = SklearnClassifier(LinearSVC())

	# # Process training data set
	classification.train(training_set)

	test_info = []

	for info in test_set:
		test_info.append(info[0])

	test_classification_index = []

	for counter, i in enumerate(themes):
		test_classification_index.append(counter)

	classification_result = classification.classify_many(test_info)

	report = classification_report(test_classification_index, classification_result, labels=list(set(test_classification_index)),target_names=themes)

	return render(request, template_name, {
		'themes': themes,
		'original_text': original_text,
		'tokens': tokens,
		'bag_of_words': bag_of_words,
		'pos_tags': pos_tags,
		'bigrams': bigrams,
		'theme_definitions': theme_definitions,
		'feature_sets': feature_sets,
		'test_set': test_set,
		'training_set': training_set,
		'classification_result': classification_result,
		'classification_report': report,
		})