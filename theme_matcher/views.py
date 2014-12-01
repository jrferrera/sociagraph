from django.shortcuts import render
from django.http import HttpResponse
from sociagraph.models import Classified_Corpus
from django.db.models import Q
# Import utilities in utils.py
from sociagraph.utils import *

def index(request):
	template_name = 'theme_matcher/index.html'

	classified_corpus = Classified_Corpus.objects.all()

	return render(request, template_name, { 'classified_corpus': classified_corpus })

def results(request):
	template_name = 'theme_matcher/results.html'

	themes = request.POST.get('theme', False)
	original_text = request.POST.get('text', False)
	
	# Process the themes
	themes = remove_non_letters(themes)
	themes = remove_extra_whitespaces(themes)
	themes = tokenize(themes)

	theme_definitions = {}
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

	# Contains all resulting data
	theme_classification_results = {}
	theme_definitions = []

	for theme in themes:
		theme_definitions.append((theme, get_word_definitions(theme)))

		labeled_text = {}

		not_theme = 'not_' + theme
		# Get text with matching theme from database
		labeled_corpora = Classified_Corpus.objects.filter(theme__contains=theme).values('text')
		
		# Get text not matching the theme from database
		opposite_labeled_corpora = Classified_Corpus.objects.filter(~Q(theme__contains=theme)).values('text')[:labeled_corpora.count()]

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
		feature_sets = get_feature_sets(combined_labeled_text, feature_set_words, theme)

		set_size = len(feature_sets)/2
		test_set = feature_sets[:set_size]
		train_set = feature_sets[set_size:]

		svm_classifier = create_svm_classifier()
		svm_classifier = train_classifier(svm_classifier, train_set)

		test_set_features = []
		test_set_correct_classifications = []

		for features, labels in test_set:
			test_set_features.append(features)
			test_set_correct_classifications.append(labels)

		test_data_classification = svm_classifier.classify_many(test_set_features)
		accuracy_score = get_accuracy_score(test_set_correct_classifications, test_data_classification)
		
		# test_paragraph = "He   adheres to the fact  that there are  several opportunities for  organic   farming .   It  can be a 'profitable  business'. Example, one can  engage in producing . organic fertilizers, organic sprays,  vermi,  and  vermicast.  The  marketing of  organic products is   a good  business  as well. Moreover  OA is  sustainable  according  to him  because  it  is  pro- environmental and  brings back  the microorganism  to the soil  that  enhances  biodiversity.  The  private  sector  according to him can also  help  sustain the promotion of OA ."
		# test_paragraph = "Focus Group Discussion (FGD) and Key Informant Interview (KII) are two important qualitative research methods. Focus Group Discussion is used to collect information from different sectors concerned in the research topic gathered in a meeting. It is a small group of six (6) to ten (10) people led through an open discussion by a skilled moderator. The group needs to be large enough to generate rich discussion but not so large that some participants are left out. It is structured around a set of carefully predetermined questions but the discussion is free-flowing where the participant comments will stimulate and influence the thinking and sharing of others. Some people even find themselves changing their thoughts and opinions during the group (Eliot and Associates, 2005). The responses will be synthesized to obtain a general idea of how these sectors perceive the issues involved in the research. This is an agriculture sector."

		classified_sentences = {}
		keywords = []

		for sentence in paragraph_to_sentences(original_text):
			classification_test = {word: (word in tokenize(sentence)) for word in feature_set_words }

			# Get each theme classification per sentence
			# classified_sentences[sentence] = svm_classifier.classify(classification_test)
			classified_sentences[sentence] = svm_classifier.classify(classification_test)

			# Get keywords from sentences matching the theme
			# if(classified_sentences[sentence].rstrip() == theme)
			# 	classified_sentences[sentence]["keywords"] = get_keywords(sentence, theme)

		theme_classification_results[theme] = {}
		theme_classification_results[theme] = classified_sentences

		test = accuracy_score

	return render(request, template_name, {
		'themes': themes,
		'theme_definitions': theme_definitions,
		'original_text': original_text,
		'tokens': tokens,
		'bag_of_words': bag_of_words,
		'pos_tags': pos_tags,
		'bigrams': bigrams,
		'theme_classification_results': theme_classification_results,
		'test': test,
		# 'testing_data': testing_data,
		# 'training_data': training_data,
		# 'classification_result': classification_result,
		# 'classification_report': classification_report,
		})

def upload_corpus(request):
	template_name = 'theme_matcher/index.html'

	filename = request.POST.get('corpus_file')
	source_file = open(filename, 'r')
	source_file.readline()
	source_file.close()
	return render(request, template_name, { 'filename': filename })