from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
    
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from elasticsearch import Elasticsearch, helpers


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
class Keyword:
    text: str
    id: Optional[int] = None



class KeywordStore(object):
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
    def _convert_hit_to_keyword(hit: dict) -> Keyword:
        """Convert a search hit into a paragraph"""
        return Keyword(
            text=hit['_source']['text'], 
            id=hit['_id']
        )

    def search_by_word_matching(
        self, 
        index: str,
        query_string: str
    ) -> List[Tuple[Keyword, float]]:
        """search by word matching

        Arguments:
            query_string {str} -- query string

        Returns:
            List[Dict] -- similar texts
        """
        query = {
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
            rep.append((KeywordStore._convert_hit_to_keyword(h), h['_score']))
        return rep


    def delete_store(self, index: str):
        """Delete store of a specified index"""
        try:
            self._es.indices.delete(index=index, ignore=[400, 404])
        except Exception as e:
            logging.warning(e)

    def add_keyword(self, index: str, keyword: Keyword):
        """Add a keyword to an es index"""
        self._es.index(index=index, body={
            'text': keyword.text,
            'as_is': keyword.text,
        }, id=keyword.id, refresh=True)

    def add_keywords(self, index: str, keywords: List[Keyword]):
        """Bulk action. Prefer this method to calling the previous add_keyword method multiple time.
        """
        actions = [{
            '_index': index,
            '_id': p.id,
            '_source': {
                'text': p.text,
                'as_is': p.text
            }
        } for p in keywords]
        helpers.bulk(self._es, actions, refresh=True)

    def update_keyword(self, index: str, keyword: Keyword):
        """Update keyword of a specified id (found in document instance). The field id therefore must not be None"""
        self._es.update(
            index=index, 
            id=keyword.id, 
            body={ 'doc': { 'text': keyword.text, 'as_is': keyword.text } },
            refresh=True
        )

    def get(self, index: str, id: int) -> Keyword:
        """Get the keyword of a given id"""
        kw = self._es.get(index=index, id=id)
        return KeywordStore._convert_hit_to_keyword(kw)

    def get_all(self, index: str) -> List[Keyword]:
        """Get all keywords in a given index"""
        res = self._es.search(index=index, body={ "size": 10000, "query": { "match_all": {} } })
        rep: List[Paragraph] = []
        for h in res["hits"]["hits"]:
            rep.append(KeywordStore._convert_hit_to_keyword(h))
        return rep