from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
    
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from elasticsearch import Elasticsearch, helpers

from .passage_ranking import Ranker


STOPWORDS_LANG_KEY = {
    "en": "_english_",
    "fr": "_french_"
}

FRENCH_FILTERS = {
    "french_keywords": {
        "type":       "keyword_marker",
        "keywords":   ["Example"] 
    },
    "french_stemmer": {
        "type":       "stemmer",
        "language":   "light_french"
    }
}

ENGLISH_FILTERS = {
    "english_stemmer": {
        "type": "stemmer",
        "language": "porter2"
    }
}

INTERROGATIVE_TERMS = {
    "en": [ "what", "where", "which", "who", "whose", "whom", "how", "def", "definition" ],
    "fr": [ "quoi", "qui", "où", "combien", "comment", "quel", "quelle", "quels", "quelles", "qu'"]
}


@dataclass
class Document:
    text: str
    id: Optional[int] = None


@dataclass
class SearchResult:
    paragraph: Document
    highlight: str
    score: float


INTERROGATIVE_TERMS = {
    "en": [ "what", "where", "which", "who", "whose", "whom", "how", "def", "definition" ],
    "fr": [ "quoi", "qui", "où", "combien", "comment", "quel", "quelle", "quels", "quelles", "qu'"]
}


class DocumentStore(object):
    """A class that wraps an Elasticsearch inside to create a search engine for a specific corpus
    """
    def __init__(self, host: str):
        self._es = Elasticsearch(hosts=[host], maxsize=10000)

    def create_store(self, index: str, lang: str = 'en', reset_index: bool = False):
        assert lang in ('en', 'fr')
        if reset_index:
            self.delete_store(index)
        if lang == "fr":
            additional_filters = FRENCH_FILTERS
            filters_list = [
                "elision",
                "lowercase",
                "stopword_filter",
                "custom_word_filter",
                "french_stemmer",
                "asciifolding",
                "trim",
                "unique"
            ]
        else: 
            additional_filters = ENGLISH_FILTERS
            filters_list = [
                "lowercase",
                "stopword_filter",
                "custom_word_filter",
                "english_stemmer",
                "asciifolding",
                "trim",
                "unique"
            ]
        self._es.indices.create(index=index, body={
            "settings": { 
                "number_of_shards": 1,
                "analysis": {
                    "filter": {
                        "stopword_filter": {
                            "type": "stop",
                            "stopwords": STOPWORDS_LANG_KEY[lang],
                            "ignore_case": True
                        },
                        "custom_word_filter": {
                            "type": "stop",
                            "stopwords": INTERROGATIVE_TERMS[lang],
                            "ignore_case": True
                        },
                        **additional_filters
                    },
                    "analyzer": {
                        "std_preprocess": {
                            "tokenizer": "standard",
                            "filter": filters_list
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "text": { "type": "text", "analyzer": "std_preprocess" }, # text field that will be analyzed
                    "as_is": { "type": "text" }
                }
            }
        })

    def suggest(
        self,
        index: str,
        query_string: str
    ) -> Tuple[List[Dict[str, Union[str, float]]], List[Dict[str, Union[str, float]]]]:
        """Suggest corrections to a query"""
        query = {
            "suggest" : {
                "text" : query_string,
                "query_suggestion" : {
                    "term" : {
                        "field" : "as_is"
                    }
                },
                "query_lemma_suggestion": {
                    "term": {
                        "field": "text"
                    }
                }
            }
        }
        res = self._es.search(index=index, body=query)
        return res['suggest']['query_suggestion'], res['suggest']['query_lemma_suggestion']

    @staticmethod
    def _convert_hit_to_document(hit: dict) -> Document:
        """Convert a search hit into a paragraph"""
        return Document(
            text=hit['_source']['text'],
            id=hit['_id']
        )

    def search(
        self, 
        index: str,
        query_string: str,
        topk: int = 3,
        ranker: Optional[Ranker] = None
    ) -> List[SearchResult]:
        """search by word matching

        Arguments:
            query_string {str} -- query string

        Returns:
            List[Dict] -- similar texts
        """
        query = {
            "size": topk,
            "query": { 
                "multi_match": {
                    "query": "*%s*" % query_string,
                    "fields": [
                        "text"
                    ]
                }
            },
            "_source": ["text"],
            "highlight": {
                "pre_tags": ["<b>"], 
                "post_tags": ["</b>"], 
                "fields": {    
                    "text": { "fragment_size": 200 }
                }
            }
        }
        
        res = self._es.search(index=index, body=query)
        rep = []
        for h in res['hits']['hits']:
            rep.append(SearchResult(
                paragraph=DocumentStore._convert_hit_to_document(h), 
                score=h['_score'],
                highlight=h["highlight"]["text"]
            ))
        if ranker:
            logging.info("Score passages...")
            scores = ranker.evaluate(query_string, [s.paragraph.text for s in rep])
            for h, sc in zip(rep, scores):
                h.score = sc
            rep.sort(key=lambda s: s.score, reverse=True)
        return rep

    def delete_store(self, index: str):
        """Delete store of a specified index"""
        try:
            self._es.indices.delete(index=index, ignore=[400, 404])
        except Exception as e:
            logging.warning(e)

    def add_document(self, index: str, document: Document):
        """Add a document to an es index"""
        self._es.index(index=index, body={
            'text': document.text,
            'as_is': document.text,
        }, id=document.id, refresh=True)

    def add_documents(self, index: str, documents: List[Document]):
        """Bulk action. Prefer this method to calling the previous add_document method multiple time.
        """
        actions = [{
            '_index': index,
            '_id': p.id,
            '_source': {
                'text': p.text,
                'as_is': p.text
            }
        } for p in documents]
        helpers.bulk(self._es, actions, refresh=True)

    def update_document(self, index: str, document: Document):
        """Update document of a specified id (found in document instance). The field id therefore must not be None"""
        self._es.update(
            index=index, 
            id=document.id, 
            body={ 'doc': { 'text': document.text, 'as_is': document.text } },
            refresh=True
        )

    def get(self, index: str, id: int) -> Document:
        """Get the document of a given id"""
        doc = self._es.get(index=index, id=id)
        return DocumentStore._convert_hit_to_document(doc)

    def index_size(self, index: str) -> int:
        """Get total number of paragraph inside the index"""
        return self._es.indices.stats(index=index)['_all']['total']['docs']['count']

    def get_all(self, index: str) -> List[Document]:
        """Get all documents in a given index"""
        res = self._es.search(index=index, body={ "size": 10000, "query": { "match_all": {} } })
        rep: List[Document] = []
        for doc in res["hits"]["hits"]:
            rep.append(DocumentStore._convert_hit_to_document(doc))
        return rep