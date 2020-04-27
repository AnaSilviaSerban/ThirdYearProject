import unittest
from unittest import TestCase

from nltk.corpus import stopwords
from SoftCosineSimilarity import SoftCosineSimilarity


class TestCosine(unittest.TestCase):
    def setUp(self):
        self.softCosineSimilarity = SoftCosineSimilarity()


class TestSoftCosineSimilarity(TestCosine):
    def test_remove_stopwords(self):
        document = ['Microsoft Software License terms\n', '\n', 'WINDOWS operating system\n', '\n',
                    'if you live in (or if your principal place of business is in) the united states, please read the binding arbitration clause and class action waiver in section 11. it affects how disputes are resolved.\n',
                    '\n', 'thank you for choosing microsoft!\n']

        stop_words = set(stopwords.words('english'))
        documents_array = []
        documents_array = self.softCosineSimilarity.remove_stopwords(documents_array, stop_words, document)
        self.assertEquals([['microsoft', 'software', 'license', 'term'], [], ['window', 'operate', 'system'], [], ['if', 'you', 'live', 'in', 'or', 'if', 'your', 'principal', 'place', 'of', 'business', 'be', 'in', 'the', 'united', 'state', 'please', 'read', 'the', 'bind', 'arbitration', 'clause', 'and', 'class', 'action', 'waiver', 'in', 'section', 'it', 'affect', 'how', 'dispute', 'be', 'resolve'], [], ['thank', 'you', 'for', 'choose', 'microsoft']], documents_array)

    def test_lowercase_words(self):
        isLower = True
        document = ['Microsoft Software License terms\n', '\n', 'WINDOWS operating system\n', '\n',
                    'if you live in (or if your principal place of business is in) the united states, please read the binding arbitration clause and class action waiver in section 11. it affects how disputes are resolved.\n',
                    '\n', 'thank you for choosing microsoft!\n']

        stop_words = set(stopwords.words('english'))
        documents_array = []
        documents_array = self.softCosineSimilarity.remove_stopwords(documents_array, stop_words, document)
        for doc in documents_array:
            for word in doc:
                if word.islower():
                    pass
                else:
                    isLower = False
                    break
        self.assertTrue(isLower)


    def test_lemmatize_words(self):
        isLemmatized = False
        document = ['Playing Microsoft Software License terms\n', '\n', 'WINDOWS operating system\n', '\n',
                    'if you live in (or if your principal place of business is in) the united states, please read the binding arbitration clause and class action waiver in section 11. it affects how disputes are resolved.\n',
                    '\n', 'thank you for choosing microsoft!\n']

        stop_words = set(stopwords.words('english'))
        documents_array = []
        documents_array = self.softCosineSimilarity.remove_stopwords(documents_array, stop_words, document)
        for doc in documents_array:
            for word in doc:
                if word == 'play':
                    isLemmatized = True

        self.assertTrue(isLemmatized)
