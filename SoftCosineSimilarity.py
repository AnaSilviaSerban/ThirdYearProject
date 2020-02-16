from allennlp.predictors import predictor
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

# connecting to the database
connection = sqlite3.connect("LegalCorpus.db")

# cursor
crsr = connection.cursor()


class SoftCosineSimilarity(object):
    # TODO try to speed up
    model = api.load("fasttext-wiki-news-subwords-300")
    # question = input()
    question = 'Who is the owner of the "Software" licensed under this Agreement?'

    def remove_stopwords(self, documents_array, stop_words, documents):
        for doc in documents:
            temp = []
            for w in simple_preprocess(doc):
                if w not in stop_words:
                    temp.append(w)
            documents_array.append(temp)

    def convert_sentences(self, sentences, dictionary, documents_array):
        for paragraph in documents_array:
            sent_doc = dictionary.doc2bow(paragraph)
            if sent_doc != []:
                sentences.append(sent_doc)

    def get_files_names(self, dir_path):
        """Iterate through directory and return a list of names of all .txt files in it."""
        return [file_name for file_name in os.listdir(dir_path) if file_name.endswith('.txt')]

    def retrieve_paragraphs(self, file_name, question_input):
        with open(file_name, encoding="ISO-8859-1") as fp:
            documents = fp.readlines()

        documents.append(question_input)
        documents = [doc.lower() for doc in documents]

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        documents_array = []
        self.remove_stopwords(documents_array, stop_words, documents)

        # Prepare a dictionary and a corpus.
        # Convert a document into a list of tokens.
        dictionary = corpora.Dictionary([word for word in documents_array])

        # Convert the sentences into bag-of-words vectors.
        sentences = []
        self.convert_sentences(sentences, dictionary, documents_array)

        # Another way of computing the similarity matrix
        similarity_index = WordEmbeddingSimilarityIndex(self.model)
        similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary)

        result_matrix = []
        result_dict = OrderedDict()

        for sentence in sentences:
            result_matrix.append(similarity_matrix.inner_product(sentences[-1], sentence, normalized=True))

        for index, result in enumerate(result_matrix):
            result_dict[index] = result

        result_dict = OrderedDict(sorted(result_dict.items(), key=itemgetter(1), reverse=True))
        para_index = list(result_dict.keys())[1]

        print(" -- FILE NAME: ", file_name)
        similarity_value = list(result_dict.values())[1]
        count = 0
        doc_paragraph = []
        for doc in documents:
            if doc != '\n':
                if count == para_index:
                    print(" -- This is the paragraph: ", similarity_value)
                    print(doc)
                    doc_paragraph = doc
                count += 1

        predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2018.11.30-charpad.tar.gz")
        prediction = predictor.predict(
            question=question_input,
            passage=doc_paragraph
        )
        print(" -- This is the span of words:")
        print(prediction["best_span_str"])
        answer = prediction["best_span_str"]
        print(type(similarity_value))

        print("------------------------")
        # SQL command to insert the data in the table
        sql_command = """INSERT INTO law_documents (filename, similarityValue, paragraphExtracted, smallAnswer) 
        VALUES (?, ?, ?, ?); """
        crsr.execute(sql_command, (file_name, float(similarity_value), doc_paragraph, answer))

"""while"""

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

