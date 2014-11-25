from django.shortcuts import render

import nltk
from sociagraph.utils import remove_non_letters
from sociagraph.utils import remove_extra_whitespaces
from sociagraph.utils import tokenize

def index(request):
	template_name = 'sentiment_analyzer/index.html'

	return render(request, template_name)

def results(request):
	template_name = 'sentiment_analyzer/results.html'

	original_text = request.POST.get('text', False)

	# Clean the text for processing
	filtered_text = remove_non_letters(original_text)
	filtered_text = remove_extra_whitespaces(filtered_text)
	tokens = tokenize(filtered_text)
	

	return render(request, template_name, {
		'filtered_text': filtered_text,
		'tokens': tokens,
		})