# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('sociagraph', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Classified_Corpus',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('text', models.TextField()),
                ('theme', models.TextField()),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='English_Dictionary',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('word', models.TextField()),
                ('associated_word', models.TextField()),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
