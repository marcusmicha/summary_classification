from typing import Any
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity

class Scorer:
    # apply dynamically the chose score
    def __init__(self, type: str, specific_metric:str) -> None:
        self.type = type
        self.specific_metric = specific_metric
        self.scorer = self.load_scorer()

    def load_scorer(self):
        if 'rouge' in self.type:
            scorer = rouge_scorer.RougeScorer([self.type]).score
            self.rouge_scorer = scorer
            return self.rouge
        else:
            return self.cosine_similarity
        
    def __call__(self, dialogue, summary) -> float:
        return self.scorer(dialogue, summary)
    
    def rouge(self, dialogue, summary) -> float:
        rouge = self.rouge_scorer(dialogue, summary)[self.type]
        return getattr(rouge, self.specific_metric)
    
    def cosine_similarity(self, x:list, y:list) -> float:
        similarity = cosine_similarity([x], [y])[0][0]
        norm_similarity = (similarity - (-1)) / (1 - (-1))
        return norm_similarity