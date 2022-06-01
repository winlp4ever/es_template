from typing import List, Dict

import boto3 


class AWSComprehend:
    def __init__(self, lang: str = 'en', **kwargs):
        """Init method"""
        self._lang = lang
        self._client = boto3.client('comprehend', region_name='eu-west-2', **kwargs)

    def entities(self, text: str) -> List[dict]:
        """Extract entities"""
        ents = self._client.detect_entities(Text=text, LanguageCode=self._lang)
        return ents['Entities']

    def keyphrases(self, text: str) -> List[dict]:
        """Extract key phrases"""
        kps = self._client.detect_key_phrases(Text=text, LanguageCode=self._lang)
        return kps['KeyPhrases']