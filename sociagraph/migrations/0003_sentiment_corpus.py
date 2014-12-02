# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('sociagraph', '0002_classified_corpus_english_dictionary'),
    ]

    operations = [
        migrations.CreateModel(
            name='Sentiment_Corpus',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('text', models.TextField()),
                ('emotion', models.TextField()),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
