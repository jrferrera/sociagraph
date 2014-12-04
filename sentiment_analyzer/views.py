from django.shortcuts import render
from django.http import HttpResponse
from sociagraph.models import Sentiment_Corpus
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from sociagraph.utils import *

def index(request):
	template_name = 'sentiment_analyzer/index.html'

	return render(request, template_name)


def corpus(request):
	template_name = 'sentiment_analyzer/corpus.html'

	# Get the corpora from the database
	sentiment_corpus = Sentiment_Corpus.objects.all()

	# Setup pagination
	paginator = Paginator(sentiment_corpus, 10)

	page = request.GET.get('page')

	try:
		corpora = paginator.page(page)
	except PageNotAnInteger:
		corpora = paginator.page(1)
	except EmptyPage:
		corpora = paginator.page(paginator.num_pages)

	return render(request, template_name, {
		'sentiment_corpus': corpora,
		'items': corpora,
		})

def results(request):
	template_name = 'sentiment_analyzer/results.html'

	original_text = request.POST.get('text', False)

	sentiments = ['happy', 'sad', 'angry', 'fearful', 'neutral']
	
	# Clean the text for processing
	filtered_text = remove_extra_whitespaces(original_text)

	# Process the tokens
	tokens = tokenize(original_text)

	# Get vocabulary size
	vocabulary_size = len(set(transform_to_text(original_text)))

	# Process the Bag of Words
	bag_of_words = get_bag_of_words(original_text)

	# Process POS Tagging
	pos_tags = get_pos_tags(original_text)
	pos_tags = get_pos_tag_values(pos_tags)

	# Get all text from database
	labeled_corpora = Sentiment_Corpus.objects.all()

	emotion_labeled_corpora = []

	for corpus in labeled_corpora:
		emotion_labeled_corpora.append((unicode_to_string(corpus.text), unicode_to_string(corpus.emotion)))

	shuffle_set(emotion_labeled_corpora)

	# Get the vocabulary
	feature_set_words = get_feature_set_words(emotion_labeled_corpora)

	# Check if the words in a paragraph is in feature set words
	feature_sets = get_feature_sets(emotion_labeled_corpora, feature_set_words)

	set_size = len(feature_sets)/2
	test_set = feature_sets[:set_size]
	train_set = feature_sets[set_size:]

	svm_classifier = create_svm_classifier()
	svm_classifier = train_classifier(svm_classifier, train_set)

	test_set_features = []
	test_set_correct_classifications = []

	# Get the test features and labels for metrics
	for features, labels in test_set:
		test_set_features.append(features)
		test_set_correct_classifications.append(labels)

	test_set_reclassification = svm_classifier.classify_many(test_set_features)
	sentiment_classification_statistics = get_classification_scores(test_set_correct_classifications, test_set_reclassification, sentiments)	

	classified_sentences = {}
	for sentence in paragraph_to_sentences(original_text):
		classification_test = {word: (word in tokenize(sentence)) for word in feature_set_words }

		# Get each theme classification per sentence
		classified_sentences[sentence] = svm_classifier.classify(classification_test)
	test = sentiment_classification_statistics

	return render(request, template_name, {
		'original_text': original_text,
		'filtered_text': filtered_text,
		'tokens': tokens,
		'pos_tags': pos_tags,
		'bag_of_words': bag_of_words,
		'classified_sentences': classified_sentences,
		'sentiment_classification_statistics': sentiment_classification_statistics,
		'test': test,
		})

def add_corpus(request):
	template_name = 'sentiment_analyzer/add_corpus.html'

	return_values = {}

	if request.POST:
		text = remove_extra_whitespaces(request.POST.get('text', False))
		emotion = remove_spaces(request.POST.get('sentiment', False)).lower()

		if text != "" or emotion != "":
			Sentiment_Corpus(text=text, emotion=emotion).save()

			return_values['notification_type'] = 'success'
			return_values['notification_message'] = 'Successfully added a corpus.'
		else:
			return_values['notification_type'] = 'error'
			return_values['notification_message'] = 'Failed to add corpus.'

	return render(request, template_name, return_values)