from django.shortcuts import render
from django.http import HttpResponse
from sociagraph.models import Sentiment_Corpus
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from sociagraph.utils import *

def index(request):
	application_name = "sentiment-analyzer"
	template_name = 'sentiment_analyzer/index.html'

	return render(request, template_name, {
		'application_name': application_name,
		})


def corpus(request):
	application_name = "sentiment-analyzer"
	template_name = 'sentiment_analyzer/corpus.html'

	# Get the corpora from the database
	sentiment_corpus = Sentiment_Corpus.objects.all()

	# Count the total corpora in the database
	total_corpora_count = sentiment_corpus.count()
	
	# Setup pagination
	paginator = Paginator(sentiment_corpus, 10)

	# Get requested page
	page = request.GET.get('page')

	try:
		corpora = paginator.page(page)
	except PageNotAnInteger:
		corpora = paginator.page(1)
	except EmptyPage:
		corpora = paginator.page(paginator.num_pages)

	new_classified_corpus = []

	for corpus in corpora:
		new_classified_corpus.append((corpus.text, corpus.emotion.split(',')))

	return render(request, template_name, {
		'application_name': application_name,
		'total_corpora_count': total_corpora_count,
		'sentiment_corpus': new_classified_corpus,
		'items': corpora,
		})

def results(request):
	application_name = "sentiment-analyzer"
	template_name = 'sentiment_analyzer/results.html'

	original_text = request.POST.get('text')

	sentiments = ['happy', 'sad', 'angry', 'fearful', 'neutral']
	
	# Clean the text for processing
	filtered_text = remove_extra_whitespaces(original_text)

	# Process the tokens
	tokens = tokenize(original_text)

	# Get vocabulary size
	vocabulary_size = get_vocabulary_count(original_text)
	original_text_length = count_words(original_text)
	# Process the Bag of Words
	bag_of_words = sort_dictionary_by_key(get_bag_of_words(original_text))

	# Process POS Tagging
	pos_tags = get_pos_tags(original_text)
	pos_tags = get_pos_tag_values(pos_tags)

	# Get all text from database
	labeled_corpora = Sentiment_Corpus.objects.all()

	# Count corpora in the database
	labeled_corpora_count = labeled_corpora.count()

	# ==== Output variables ====
	corpora_statistics = {}
	emotion_labeled_corpora = []
	overall_sentiment = ''
	# ==========================

	for corpus in labeled_corpora:
		emotion_labeled_corpora.append((unicode_to_string(remove_stopwords(corpus.text)), unicode_to_string(corpus.emotion)))

	shuffle_set(emotion_labeled_corpora)

	# Get the vocabulary
	feature_set_words = get_feature_set_words(emotion_labeled_corpora)

	# Check if the words in a paragraph is in feature set words
	feature_sets = get_sentiment_feature_sets(emotion_labeled_corpora, feature_set_words, sentiments)

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
	sentiment_frequency = { 'happy': 0, 'sad': 0, 'angry': 0, 'fearful': 0, 'neutral': 0 }

	for sentence in paragraph_to_sentences(original_text):
		classification_test = get_sentiment_features(sentence, feature_set_words, sentiments)

		# Get each theme classification per sentence
		classification = svm_classifier.classify(classification_test)
		classified_sentences[sentence] = classification
		sentiment_frequency.update({ classification: sentiment_frequency[classification]+1 })
	overall_sentiment = get_most_frequent_sentiment(sentiment_frequency)
	corpora_statistics = sort_dictionary_by_key({ 'Corpora Total': labeled_corpora_count, 'Test Set Count': labeled_corpora_count/2, 'Train Set Count': labeled_corpora_count /2})
	sentiment_frequency = sort_dictionary_by_value(sentiment_frequency)
	return render(request, template_name, {
		'application_name': application_name,
		'original_text': original_text,
		'filtered_text': filtered_text,
		'vocabulary_size': vocabulary_size,
		'original_text_length': original_text_length,
		'corpora_statistics': corpora_statistics,
		'tokens': tokens,
		'pos_tags': pos_tags,
		'bag_of_words': bag_of_words,
		'sentiment_frequency': sentiment_frequency,
		'classified_sentences': classified_sentences,
		'overall_sentiment': overall_sentiment,
		'sentiment_classification_statistics': sentiment_classification_statistics,
		})

def add_corpus(request):
	application_name = "sentiment-analyzer"
	template_name = 'sentiment_analyzer/add_corpus.html'

	return_values = {}
	return_values['application_name'] = application_name

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