#%%

from pycaret.datasets import get_data
# %%
data = get_data ('kiva')
#%%
import pandas as pd
df_corpus = pd.read_excel("../../storage/ClientesBotonPagos__202006.xlsx")
# %%
df_kaggle_train = pd.read_csv("../../storage/kaggle_train.csv")

df_kaggle_test = pd.read_csv("../../storage/kaggle_test.csv")

df_kaggle_sample = pd.read_csv("../../storage/kaggle_samplesub.csv")
# %%

df_corpus.describe()
# %%
df_corpus.head()
# %%
df_corpus.NOMBRE_COMERCIO.head()
# %%
from nltk.corpus import stopwords
stop_words = stopwords.words('spanish')

from stop_words import get_stop_words
stop_words = get_stop_words('spanish')
# %%
stop_words
# %%
from pycaret.nlp import *
nlp1 = setup(df_corpus, target = 'NOMBRE_COMERCIO', custom_stopwords = stop_words, session_id = 21, log_experiment= True, experiment_name = 'ClientesBotonPagos')
# %%

# %%

# %%

# %%
from pycaret.nlp import *
nlp1 = setup(data, target = 'es', log_experiment = True, experiment_name = 'kiva1')
# %%
lda = create_model('lda')
# %%
nmf = create_model('nmf')
# %%
!mlflow ui
# %%
# %%
# Start with loading all necessary libraries
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

def text_preprocess(df):
    print(df.head())
    df.question_text = df.question_text.apply(lambda x: x.lower())
    print (df.head())
    import re
    df.question_text = df.question_text.apply(lambda x: re.sub(r'\d+', '', x) )
    print (df.head())   
    import string
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    df.question_text = df.question_text.apply(lambda x: regex.sub('', x))
    print (df.head())
    df.question_text = df.question_text.apply(lambda x: x.strip())
    print (df.head(20))

#%%

def main():
    """ this is the main repository for display an Bag of words """
    #sample of one register
    aux_String = ""
    for iNombre in df_corpus.NOMBRE_COMERCIO:
        aux_String += iNombre
    text = aux_String
    print ("there are {} words in all questions".format(len(text)))
    stopwords = set(stop_words)
    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

#%%
main()
# %%


#%%
df_kaggle_train.describe()
# %%
df_kaggle_train.head()
# %%
df_kg_cli = df_kaggle_train.merge(df_corpus, how = 'left', left_on  = 'num_doc', right_on = 'NUMERO_DOCUMENTO')
# %%
df_kg_cli_test = pd.merge(df_kaggle_test, df_corpus, left_on  = 'num_doc', right_on = 'NUMERO_DOCUMENTO')

# %%
df_kg_cli.head()
# %%
from pycaret.classification import *
pycar = setup(df_kaggle_train, target= 'flag_multi')
# %%
compare_models()
# %%

lda = create_model('lda')
# %%
tuned_lda = tune_model(lda,optimize='AUC')
# %%
pred = predict_model(lda,df_kaggle_test)
# %%
output = pd.merge(pred.id, pred.Label, on = pred.id.index)
# %%
df_output = pd.DataFrame(output,columns = ['id','Label'])
df_output.columns = ['id','flag_multi']
# %%
df_output.to_csv("output.csv",index= False)
# %%
pivot = pd.pivot_table(df_kaggle_train, index = ['flag_multi','marca_nompen','tipo_doc'], aggfunc=len)
# %%
type(pivot)
# %%
pivot = pivot.reset_index()
# %%
res = pd.merge(pivot.id[pivot.flag_multi==0], pivot.id[pivot.flag_multi==1])
# %%

# %%

# %%
