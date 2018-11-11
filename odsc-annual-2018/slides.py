import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')
sns.set_style('ticks')
plt.rcParams['figure.figsize'] = (12, 6)

'''
*Non-standard problems in feature engineering*

In this talk we focus on three problems:

- Good exploratory analysis (IBM data set)
- Missing data
- Keep an eye on the distributions

What is out:
- Modeling and selection
- Unstructured data, e.g., images, text

Exploratory plots/analysis designed to get a sneek peek into key features highly
correlated with the response. Hints about non-linearities are highlighted at the
outset

Good methods that can often kickstart modeling:
- pairwise plots
- simple calculation and plots of correlations with the target
- projections of data into a lower dimensional space

'''

