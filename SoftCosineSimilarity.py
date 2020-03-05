from gensim import corpora
import gensim.downloader as api
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from collections import OrderedDict
from operator import itemgetter
from allennlp.predictors.predictor import Predictor
import os
import sqlite3
from nltk.stem import WordNetLemmatizer

# connecting to the database
connection = sqlite3.connect("LegalCorpus.db")

# cursor
crsr = connection.cursor()


# model = api.load("fasttext-wiki-news-subwords-300")
# index = WordEmbeddingSimilarityIndex(model)
# filename_similarity = 'sparse_model.pkl'
# index.save(filename_similarity)


class SoftCosineSimilarity(object):
    # TODO try to speed up
    # Lemmatize and tokenize the words (simple_preprocess)
    # Loops through the lists in the document and tokenizes the word
    # while removing stopwords
    def remove_stopwords(self, documents_array, stop_words, documents):
        lemmatizer = WordNetLemmatizer()
        for doc in documents:
            temp = []
            for w in simple_preprocess(doc):
                if w not in stop_words:
                    temp.append(lemmatizer.lemmatize(w))
            documents_array.append(temp)

    """Convert `paragraphs` into the bag-of-words (BoW) format = list of `(token_id, token_count)` tuples."""

    def convert_sentences(self, sentences, dictionary, documents_array):
        for paragraph in documents_array:
            sent_doc = dictionary.doc2bow(paragraph)
            if sent_doc != []:
                sentences.append(sent_doc)

    """Iterate through directory and return a list of names of all .txt files in it."""

    def get_files_names(self, dir_path):
        return [file_name for file_name in os.listdir(dir_path) if file_name.endswith('.txt')]

    def retrieve_paragraphs(self, file_name, question_input):
        # Reading file names and read them paragraph by paragraph.
        with open(file_name, "r") as fp:
            documents = fp.readlines()

        # Append question to the end of the document and pre-process everything all at once.
        documents.append(question_input)

        # convert everything to lower case for better processing.
        documents = [doc.lower() for doc in documents]

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        documents_array = []
        self.remove_stopwords(documents_array, stop_words, documents)

        # Prepare a dictionary and a corpus.
        # Convert a document into a list of tokens.
        # This module implements the concept of a Dictionary -- a mapping between paragraphs and their integer ids.
        dictionary = corpora.Dictionary([word for word in documents_array])

        # Convert the sentences into bag-of-words vectors.
        sentences = []
        self.convert_sentences(sentences, dictionary, documents_array)

        # Another way of computing the similarity matrix
        # similarity_index = WordEmbeddingSimilarityIndex(self.loaded_model)
        similarity_index = WordEmbeddingSimilarityIndex.load('sparse_model.pkl')
        similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary)

        result_matrix = []
        result_dict = OrderedDict()

        for sentence in sentences:
            result_matrix.append(similarity_matrix.inner_product(sentences[-1], sentence, normalized=True))

        # print(" -- Similarity between document 1 and document 2: -- ")
        # for sentence in sentences:
        #     print(rank(sentences[-1], sentence))
        #     print(sentence)
        # print(" ------ ")

        for index, result in enumerate(result_matrix):
            result_dict[index] = result

        result_dict = OrderedDict(sorted(result_dict.items(), key=itemgetter(1), reverse=True))
        para_index = [list(result_dict.keys())[1], list(result_dict.keys())[2], list(result_dict.keys())[3],
                      list(result_dict.keys())[4]]

        print(" -- FILE NAME: ", file_name)
        similarity_value = list(result_dict.values())[1]
        count = 0
        doc_paragraph = ''
        for doc in documents:
            if doc != '\n':
                if count in para_index:
                    print(" -- This is the paragraph: ", similarity_value)
                    print(doc)
                    doc_paragraph = doc_paragraph + ' ' + doc
                count += 1

        answer = "There is no answer for this document!"
        print(" -- This is the span of words:")
        if similarity_value > 0.3:
            predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2018.11.30-charpad.tar.gz")
            prediction = predictor.predict(
                question=question_input,
                passage=doc_paragraph
            )
            print(prediction["best_span_str"])
            answer = prediction["best_span_str"]
        else:
            print(answer)
        print("------------------------")
        # SQL command to insert the data in the table
        sql_command = """INSERT INTO law_documents (filename, similarityValue, paragraphExtracted, smallAnswer) 
        VALUES (?, ?, ?, ?); """

        crsr.execute(sql_command, (file_name, float(similarity_value), doc_paragraph, answer))


"""while true"""

softCosineSimilarity = SoftCosineSimilarity()

while True:
    crsr.execute("DELETE FROM law_documents")
    question = input("Please ask a question: ")
    file_name_list = softCosineSimilarity.get_files_names('/Users/serbana/PycharmProjects/ThirdYearProject')
    print(file_name_list)
    for file in file_name_list:
        softCosineSimilarity.retrieve_paragraphs(file, question)

    crsr.execute("SELECT * FROM law_documents order by similarityValue desc limit 3")
    ans = crsr.fetchall()

    # loop to print all the data
    for i in ans:
        print(i)

    # close the connection
