$(document).ready(function() {
	$('.notification').fadeOut(5000);

	$("#key-information-extractor-nav a").mouseenter(function() {
		$("#sentiment-analyzer-nav").removeClass("open");
		$("#key-information-extractor-nav").addClass("open");
	});

	$("#sentiment-analyzer-nav a").mouseenter(function() {
		$("#key-information-extractor-nav").removeClass("open");
		$("#sentiment-analyzer-nav").addClass("open");
	});
});