from django.db import models

class POS_Tags(models.Model):
	tag = models.CharField(max_length=10)
	tag_value = models.CharField(max_length=300)