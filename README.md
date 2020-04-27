The library requirements for this project are:
 - gensim
 - nltk
 - collections
 - allennlp
 - fasttext-wiki-news-subwords-300

The project was run from PyCharm and need to set the directory from where the project is run.

When the algorithm is run for the forst time, the following lines need to be uncommented in order to create the sparse_model.pkl file
where I saved the pre-trained word embeddings in order to not load them every time the project runs:
# model = api.load("fasttext-wiki-news-subwords-300")
# index = WordEmbeddingSimilarityIndex(model)
# filename_similarity = 'sparse_model.pkl'
# index.save(filename_similarity)

The txt files need to be in the same folder as the py file with the project code.




