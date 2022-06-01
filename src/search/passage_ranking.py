from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
    
import logging
logging.basicConfig(level='INFO')
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
import time

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import torch
import regex
from sentence_transformers import CrossEncoder
import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset
import numpy as np
from tqdm import tqdm



class Ranker(ABC):
    def __init__(self):
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @abstractmethod
    def evaluate(self, question: str, answers: List[str]):
        """ Evaluate question answers compatibility by outputing for each answer, a proba
        between 0 and 1 indicating whether it is a correct answer of the given question """
        pass
        

class ResponseScorer(Ranker):
    MODELS = {
        'en': {
            "name": "models/asnq_roberta",
            "tokenizer": "models/asnq_roberta",
        }
    }
    
    def __init__(self, lang='en'):
        """Init function

        Keyword Arguments:
            lang {str} -- model language (default: {'en'})
        """
        super().__init__()
        assert lang in ResponseScorer.MODELS, "Language %s is not supported yet." % lang
        self._tokenizer = RobertaTokenizerFast.from_pretrained(ResponseScorer.MODELS[lang]["tokenizer"], max_length=512)
        self._model = RobertaForSequenceClassification.from_pretrained(ResponseScorer.MODELS[lang]["name"]).to(self._device)

    def infer(self, qas: List[Tuple[str, str]], batch_size: int = 16, max_length: int = 512) -> List[float]:
        """Score a list of pairs of question-answer"""
        try:
            st = time.time()
            toks = self._tokenizer([(q, a) for q, a in qas], truncation=True, padding=True, max_length=max_length)
            data = Dataset.from_dict({ 'q': [u[0] for u in qas], 'a': [u[1] for u in qas], **toks })            
            data.set_format(type='torch', columns=['input_ids', 'attention_mask'])
            loader = DataLoader(data, batch_size=batch_size)
            logging.info("Prepare data: {:.2f}s".format(time.time()-st))

            probas = []
            st = time.time()
            self._model.eval()
            with torch.no_grad():
                for i, batch in enumerate(tqdm(loader)):
                    batch = { k: v.to(self._device) for k, v in batch.items() }
                    outputs = self._model(**batch)
                    ps = torch.nn.functional.softmax(outputs.logits, dim=1)[:,1]
                    probas.append(ps)
            # if using gpu, empty gpu memory cache after inferences
            probas = torch.cat(probas).cpu().tolist()
            logging.info("Inference: {:.2f}".format(time.time()-st))
            return probas
        except Exception as e:
            raise e
        finally:
            if self._device == 'cuda':
                torch.cuda.empty_cache()

    def evaluate(self, question: str, answers: List[str], batch_size: int = 16, max_length: int = 512) -> List[float]:
        """Calculate compatibility scores between a question and a list of answers

        Arguments:
            question {str} -- a question
            answers {List[str]} -- a list of answers

        Keyword Arguments:
            batch_size {int} -- batch size used in the model (default: {16})

        Raises:
            e: [description]

        Returns:
            List[float] -- list of probabilities that corresponds to the answers
        """
        return self.infer([(question, a) for a in answers], batch_size, max_length)


class STRanker(Ranker):
    def __init__(self, model_name='nboost/pt-bert-large-msmarco'):
        self._scorer = CrossEncoder(model_name, max_length=512)

    def evaluate(self, question: str, answers: List[str]) -> List[float]:
        scores = self._scorer.predict([(question, ans) for ans in answers])
        return scores