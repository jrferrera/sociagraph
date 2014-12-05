from django.shortcuts import render
from django.http import HttpResponse
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

from sociagraph.models import Classified_Corpus
from django.db.models import Q

from sociagraph.utils import *


def index(request):
	template_name = 'theme_matcher/index.html'

	return render(request, template_name)


def corpus(request):
	template_name = 'theme_matcher/corpus.html'

	# Get the corpora from the database
	classified_corpus = Classified_Corpus.objects.all()

	# Setup pagination
	paginator = Paginator(classified_corpus, 10)

	page = request.GET.get('page')

	try:
		corpora = paginator.page(page)
	except PageNotAnInteger:
		corpora = paginator.page(1)
	except EmptyPage:
		corpora = paginator.page(paginator.num_pages)

	return render(request, template_name, {
		'classified_corpus': corpora,
		'items': corpora,
		})


def results(request):
	template_name = 'theme_matcher/results.html'

	themes = request.POST.get('theme')
	original_text = request.POST.get('text')
	
	# Clean the themes
	themes = remove_non_letters(themes)
	themes = remove_extra_whitespaces(themes)
	theme_list = tokenize(themes)

	# ==== Output variables ====
	tokens = tokenize(original_text)
	original_text_length = count_words(original_text)
	vocabulary_size = get_vocabulary_count(original_text)
	bag_of_words = sort_dictionary_by_key(get_bag_of_words(original_text))

	pos_tags = get_pos_tags(original_text)
	pos_tags = get_pos_tag_values(pos_tags)

	theme_classification_results = {}
	theme_classification_statistics = {}
	theme_definitions = {}
	corpora_statistics = {}
	# ==========================

	test = ''

	for theme in theme_list:
		theme_definitions[theme] = get_word_definitions(theme)

		labeled_text = {}

		not_theme = 'not_' + theme

		# Get text with matching theme from database
		# labeled_corpora = Classified_Corpus.objects.filter(theme__contains=theme).values('text')
		labeled_corpora = Classified_Corpus.objects.filter(theme__contains=theme).values('text')
		
		labeled_corpora_count = labeled_corpora.count()

		if labeled_corpora_count >= 3:
			# Get text not matching the theme from database
			opposite_labeled_corpora = Classified_Corpus.objects.filter(~Q(theme__contains=theme)).values('text').order_by('?')[:labeled_corpora.count()]

			# Assign each result to given theme
			labeled_text[theme] = assign_theme(labeled_corpora, theme)

			# Assign each result to given theme
			labeled_text[not_theme] = assign_theme(opposite_labeled_corpora, not_theme)

			# Combine opposing themes
			combined_labeled_text = labeled_text[theme] + labeled_text[not_theme]
			
			# Shuffle the combined text with labels
			shuffle_set(combined_labeled_text)
			
			# Get the vocabulary of the combined labels
			feature_set_words = get_feature_set_words(combined_labeled_text)

			# Check if the words in a paragraph is in feature set words
			feature_sets = get_theme_corpus_feature_sets(combined_labeled_text, feature_set_words, theme)

			set_size = len(feature_sets)/2
			test_set = feature_sets[:set_size]
			train_set = feature_sets[set_size:]

			svm_classifier = create_svm_classifier()
			svm_classifier = train_classifier(svm_classifier, train_set)

			test_set_features = []
			test_set_correct_classifications = []
			classification_scores = {}

			# Get the test features and labels for metrics
			for features, labels in test_set:
				test_set_features.append(features)
				test_set_correct_classifications.append(labels)

			test_set_reclassification = svm_classifier.classify_many(test_set_features)
			classification_scores = get_classification_scores(test_set_correct_classifications, test_set_reclassification, [theme, not_theme])

			classified_sentences = {}
			keywords = []

			for sentence in paragraph_to_sentences(original_text):
				# classification_test = { word: (word in tokenize(sentence)) for word in feature_set_words }
				classification_test = get_features(sentence, feature_set_words, theme)

				# Get each theme classification per sentence
				classified_sentences[sentence] = svm_classifier.classify(classification_test)

				# Get keywords from sentences matching the theme
				# if(classified_sentences[sentence].rstrip() == theme)
				# 	classified_sentences[sentence]["keywords"] = get_keywords(sentence, theme)

			theme_classification_results[theme] = classified_sentences
			theme_classification_statistics[theme] = classification_scores
			corpora_statistics[theme] = sort_dictionary_by_key({ 'Corpora Total': labeled_corpora_count, 'Test Set Count': labeled_corpora_count, 'Train Set Count': labeled_corpora_count })
		else:
			theme_classification_results[theme] = None
			theme_classification_statistics[theme] = None
			corpora_statistics[theme] = None

		# test = test_set

	return render(request, template_name, {
		'theme_definitions': theme_definitions,
		'original_text': original_text,
		'vocabulary_size': vocabulary_size,
		'original_text_length': original_text_length,
		'tokens': tokens,
		'bag_of_words': bag_of_words,
		'pos_tags': pos_tags,
		'theme_classification_results': theme_classification_results,
		'theme_classification_statistics': theme_classification_statistics,
		'corpora_statistics': corpora_statistics,
		'test': test,
		})


def add_corpus(request):
	template_name = 'theme_matcher/add_corpus.html'

	return_values = {}

	if request.POST:
		text = remove_extra_whitespaces(request.POST.get('text', False))
		theme = remove_spaces(request.POST.get('theme', False)).lower()

		if text != "" or theme != "":
			Classified_Corpus(text=text, theme=theme).save()

			return_values['notification_type'] = 'success'
			return_values['notification_message'] = 'Successfully added a corpus.'
		else:
			return_values['notification_type'] = 'error'
			return_values['notification_message'] = 'Failed to add corpus.'

	return render(request, template_name, return_values)