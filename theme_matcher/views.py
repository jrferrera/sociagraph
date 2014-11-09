from django.shortcuts import render
from django.http import HttpResponse

# Import utilities in utils.py
from theme_matcher.utils import remove_non_letters
from theme_matcher.utils import remove_extra_whitespaces
from theme_matcher.utils import tokenize
from theme_matcher.utils import get_bag_of_words
from theme_matcher.utils import get_pos_tags
from theme_matcher.utils import get_pos_tag_values
from theme_matcher.utils import get_bigrams

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

	return render(request, template_name, {
		'themes': themes,
		'original_text': original_text,
		'tokens': tokens,
		'bag_of_words': bag_of_words,
		'pos_tags': pos_tags,
		'bigrams': bigrams,
		})