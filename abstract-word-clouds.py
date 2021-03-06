import random

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(49, 51)

papers_data_df = pd.read_csv('papers-list.csv')
stopwords = set(STOPWORDS)

for additional_word in ['data', 'visualization', 'visualizations', 'visualize']:
    stopwords.add(additional_word)

papers_data_df = papers_data_df.loc[papers_data_df.Year >= 2009, :]


for year in [2009, 2018]:
    subset_data_df = papers_data_df.loc[papers_data_df.Year == year]
    abstracts = '\n'.join(subset_data_df.Abstract)
    
    wc = WordCloud(background_color="white",
                   collocations=False,
                   width=600,
                   height=400,
                   max_words=50, 
                   stopwords=stopwords, 
                   max_font_size=40, 
                   random_state=20130810)
    
    wc.generate(abstracts.lower())

    plt.figure(figsize=(20,10))
    
    plt.imshow(wc.recolor(color_func=grey_color_func, random_state=20130810),
               interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
    wc.to_file(str(year)+'.jpg')
