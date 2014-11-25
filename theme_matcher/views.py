from django.shortcuts import render
from django.http import HttpResponse

# Import utilities in utils.py
import nltk
from sociagraph.utils import remove_non_letters
from sociagraph.utils import remove_extra_whitespaces
from sociagraph.utils import remove_stopwords
from sociagraph.utils import tokenize
from sociagraph.utils import get_bag_of_words
from sociagraph.utils import get_pos_tags
from sociagraph.utils import get_pos_tag_values
from sociagraph.utils import get_bigrams
from sociagraph.utils import in_dictionary
from sociagraph.utils import get_synonyms
from sociagraph.utils import get_word_definitions
from sociagraph.utils import get_pos_tag
from sociagraph.utils import get_pos_tag_value
from sociagraph.utils import get_initial_classifications
from sociagraph.utils import build_feature_sets
from sociagraph.utils import has_similar_synonyms
from sociagraph.utils import shuffle_set
from sociagraph.utils import create_svm_classifier
from sociagraph.utils import train_classifier
from sociagraph.utils import get_classification_report
from sociagraph.utils import unicode_to_string
from sociagraph.utils import *

def index(request):
	template_name = 'theme_matcher/index.html'

	return render(request, template_name)

def results(request):
	template_name = 'theme_matcher/results.html'

	themes = request.POST.get('theme', False)
	original_text = request.POST.get('text', False)
	
	# Process the themes
	themes = remove_non_letters(themes)
	themes = remove_extra_whitespaces(themes)
	themes = tokenize(themes)

	# Process the tokens
	tokens = tokenize(original_text)

	# === Analyze the contents ==
	original_text_length = count_words(original_text)

	# Get vocabulary size
	vocabulary_size = len(set(transform_to_text(original_text)))
	# ===========================

	# Process the Bag of Words
	bag_of_words = get_bag_of_words(original_text)

	# Process POS Tagging
	pos_tags = get_pos_tags(original_text)
	pos_tags = get_pos_tag_values(pos_tags)

	# Process bigrams
	bigrams = get_bigrams(original_text)

	# ========================================
	# Get from database
	initial_training_data = [
					("It  was in 1999 when Mr. Aquino was introduced to Organic  Agriculture while he was  still  with the military as a  soldier (sundalo )", "agriculture"),
					("It  was in 1999 when Mr. Aquino was introduced to Organic  Agriculture while he was  still  with the military as a  soldier (sundalo )", "organic"),
					("It  was in 1999 when Mr. Aquino was introduced to Organic  Agriculture while he was  still  with the military as a  soldier (sundalo )", "military"),
					("It  was in 1999 when Mr. Aquino was introduced to Organic  Agriculture while he was  still  with the military as a  soldier (sundalo )", "person"),
					("He  is proud to  serve the  country  as a military soldier  for   26 years and  9 months .  Although he  finished  BA political Science , he  was  interested  in farming . As a   soldier , he   farm  for pleasure.", "military"),
					("He  is proud to  serve the  country  as a military soldier  for   26 years and  9 months .  Although he  finished  BA political Science , he  was  interested  in farming . As a   soldier , he   farm  for pleasure.", "education"),
					("He  is proud to  serve the  country  as a military soldier  for   26 years and  9 months .  Although he  finished  BA political Science , he  was  interested  in farming . As a   soldier , he   farm  for pleasure.", "agriculture"),

				   ]
	# ========================================
	sentence_list = paragraph_to_sentences(original_text)
	classified_sentences = get_initial_sentence_classification(themes, sentence_list)

	# Append the initial sentence classification to initial training data
	for classified_sentence in classified_sentences:
		# Append the sentence and label
		initial_training_data.append((classified_sentence[0], classified_sentence[1]))

	# Get training data vocabulary
	training_data_words = set(word.lower() for information in initial_training_data for word in tokenize(information[0]))

	training_data = [ ({ word: (word.lower() in tokenize(information[0])) for word in training_data_words }, information[1]) for information in initial_training_data ]

	svm_classifier = create_svm_classifier()
	svm_classifier = train_classifier(svm_classifier, training_data)

	test_sent_features = {word.lower(): (word in tokenize(original_text.lower())) for word in training_data_words}

	test = svm_classifier.classify(test_sent_features)
	# test = training_data
	
	return render(request, template_name, {
		'themes': themes,
		'original_text': original_text,
		'tokens': tokens,
		'bag_of_words': bag_of_words,
		'pos_tags': pos_tags,
		'bigrams': bigrams,
		'sentence_list': sentence_list,
		'classified_sentences': classified_sentences,
		'test': test,
		# 'feature_sets': feature_sets,
		# 'testing_data': testing_data,
		# 'training_data': training_data,
		# 'classification_result': classification_result,
		# 'classification_report': classification_report,
		})