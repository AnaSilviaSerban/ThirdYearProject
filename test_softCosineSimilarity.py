import unittest
from unittest import TestCase

from nltk.corpus import stopwords
from SoftCosineSimilarity import SoftCosineSimilarity


class TestCosine(unittest.TestCase):
    def setUp(self):
        self.softCosineSimilarity = SoftCosineSimilarity()


class TestSoftCosineSimilarity(TestCosine):
    def test_remove_stopwords(self):
        document = "These words are said to have a very low discrimination value when it comes to IR and they are " \
                   "known as stopwords or sometimes as noise words or the negative dictionary. "

        stop_words = set(stopwords.words('english'))
        documents_array = []
        self.softCosineSimilarity.remove_stopwords(documents_array, stop_words, document)
        self.assertNotEquals(document, documents_array)

    def test_convert_sentences(self):
        self.fail()

    def test_retrieve_paragraphs(self):
        self.fail()
