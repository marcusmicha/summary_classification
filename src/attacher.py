from src.scorer import Scorer
import pandas as pd
import numpy as np
import re
import os.path
import math
import itertools
import sys
import munkres
from tqdm.auto import tqdm
tqdm.pandas()
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rich.console import Console
console = Console()

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]+"
)

class Attacher:
    def __init__(self, dialogues_path, summaries_path, score_details, **config) -> None:
        self.config = config
        self.dialogues = self.load_and_clean(dialogues_path, ['dialogue'])
        self.summaries = self.load_and_clean(summaries_path, ['summary_piece'])
        self.score_details = score_details
        self.embedding_model = SentenceTransformer(self.config['embeddings_model'])
        self.attaching_strategy = self.prepare_attach()
        self.attached_df = pd.DataFrame()
        self.sorted_df = pd.DataFrame()

    def load_and_clean(self, csv_path:str, columns:list[str]) -> pd.DataFrame:
        # Simple data cleaning procedure
        df = pd.read_csv(csv_path)
        for column in columns:
            df[column] = df[column].str.replace('\r\n', '. ') # Replacing newline characters with dot
            df[column] = df[column].str.replace(r'([?!\.]+)(\.)', r'\1', n=1, regex=True) # Removing unnecessary dots
            df[column] = df[column].str.replace(EMOJI_PATTERN, '', regex=True) # Removing Emojis
            df[column] = df[column].str.replace(r'\s{2,}', ' ', regex=True) # removing multiple space
        return df
    
    def prepare_attach(self) -> dict:
        # prepare scoring strategy
        if 'rouge' in self.score_details['type']:
            attaching_startegy = {
                **self.score_details,
                'dialogue_col':'dialogue',
                'summary_col': 'summary_piece'
                }
        else:
            # cosine_similarity strategy
            attaching_startegy = self.apply_embeddings()
        return attaching_startegy
    
    def apply_embeddings(self, embeddings_col:str = 'embeddings') -> dict:
        # summarize dialogues
        console.print('[bold cyan] -- Summarizing dialogues -- [/bold cyan]')
        decode = self.summarize_dialogues()
        self.dialogues['summary'] = self.dialogues.progress_apply(lambda row: decode(row['dialogue']), axis=1)
        dialogues = self.dialogues.summary.values.tolist()
        embeddings_d = self.embedding_model.encode(dialogues)
        summaries = self.summaries.summary_piece.values.tolist()
        embeddings_s = self.embedding_model.encode(summaries)

        self.dialogues[embeddings_col] = embeddings_d.tolist()
        self.summaries[embeddings_col] = embeddings_s.tolist()
        attaching_strategy = {
                **self.score_details,
                'dialogue_col': embeddings_col,
                'summary_col': embeddings_col
                }
        return attaching_strategy
    
    def summarize_dialogues(self):
        summarizer_model = self.config['summarizer_model']
        tokenizer = T5Tokenizer.from_pretrained(summarizer_model['name'])
        model = T5ForConditionalGeneration.from_pretrained(summarizer_model['name'])

        def decode(text: str, tokenizer=tokenizer, model=model) -> str:
            inputs = tokenizer.batch_encode_plus([summarizer_model['prefix'] + text], max_length=summarizer_model['max_input_length'], return_tensors="pt", padding='longest')  # Batch size 1
            outputs = model.generate(inputs['input_ids'], num_beams=4, max_length=summarizer_model['max_output_length'], early_stopping=True)
            return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in outputs][0]
        return decode

    
    def __call__(self) -> pd.DataFrame:
        score_matrix = self.build_score_matrix()
        attached_df = self.attach_summaries(score_matrix)
        return attached_df
        
    def build_score_matrix(self) -> pd.DataFrame:
        # Check if the matrix was already calculated and load it in that case
        if os.path.exists(self.config['score_matrix_path']):
            console.print('[bold cyan] • Loading score matrix [/bold cyan]')
            return np.loadtxt(self.config['score_matrix_path'])
        
        # We create a matrix to store the proportion of entities from summaries present in dialogues
        score_matrix = np.zeros((len(self.summaries), len(self.dialogues)))
        scorer = Scorer(**self.score_details)

        # Iterate through each row in summaries and dialogues
        console.print('[bold cyan] -- Scoring running -- [/bold cyan]')
        for i, summary in enumerate(tqdm(self.summaries[self.attaching_strategy['summary_col']])):
            for j, text in enumerate(self.dialogues[self.attaching_strategy['dialogue_col']]):
                score_matrix[i, j] = scorer(text, summary)
        np.savetxt(self.config['score_matrix_path'], score_matrix)
        return score_matrix

    def attach_summaries(self, score_matrix):
        # Keep best dialogue for each summary base the specified_score
        most_similar_indices = np.argmax(score_matrix, axis=1)
        scores = np.max(score_matrix, axis=1)

        # Save best dialogues id
        self.summaries['dialogue_id'] = most_similar_indices
        self.summaries['score'] = scores

        # Aggergate results
        attached_df = pd.merge(self.dialogues, self.summaries, left_index=True, right_on='dialogue_id')[['dialogue', 'summary_piece','dialogue_id']]
        attached_df = attached_df.groupby('dialogue', as_index=False)['summary_piece'].apply(list)
        self.attached_df = attached_df
        return self.attached_df 
    
    def sort(self) -> pd.DataFrame:
        try:
            assert not self.attached_df.empty
        except AssertionError:
           console.print('[red] Text and summary not attached yet [/red]')
           return
        df = self.attached_df.copy()
        df['splitted_dialogue'] = df.apply(lambda row: self.order_summary(row['dialogue'], row['summary_piece']), axis=1)
        df['nb_summaries'] = df.summary_piece.apply(len)
        df = df[df['nb_summaries'] < 10]
        console.print('[bold cyan] -- Sorting summaries -- [/bold cyan]')
        df['ordered_summary_piece'] = df.progress_apply(lambda row: self.dual_similarity(row['splitted_dialogue'], row['summary_piece']), axis=1)
        df['final_summary'] = df.summary_piece.apply(' '.join)
        self.sorted_df = self.retrieve_ids(df)
        return self.sorted_df

    def order_summary(self, text, summaries):
        nb_summaries = len(summaries)
        splits = self.split_text(text, nb_summaries)
        return splits
    
    def split_text(self, text: str, n: int) -> list:
        step = math.ceil(len(text)/n)
        return [text[i:i + step] for i in range(0, len(text), step)]
    
    def dual_similarity(self, dialogues, summaries):
        embeddings_d = self.embedding_model.encode(dialogues)
        embeddings_s = self.embedding_model.encode(summaries)

        # Create the similarity_matrix
        similarity_matrix = cosine_similarity(embeddings_s, embeddings_d)
        # most_similar_indices = np.argmax(similarity_matrix, axis=1)
        inds = self.maximize_trace(similarity_matrix)
        inds.sort(key=lambda x: x[1])
        ordered_summaries = [summaries[i[0]] for i in inds]
        return ordered_summaries
    
    def maximize_trace(self, mat):
        """
        Maximize trace by minimizing the Frobenius norm of 
        `np.dot(p, mat)-np.eye(mat.shape[0])`, where `mat` is square and
        `p` is a permutation matrix. Returns permuted version of `mat` with
        maximal trace.
        """

        assert mat.shape[0] == mat.shape[1]
        d = np.zeros_like(mat)
        n = mat.shape[0]
        b = np.eye(n, dtype=int)
        for i, j in itertools.product(range(n), range(n)):
            d[j, i] = sum((b[j, :]-mat[i, :])**2)
        m = munkres.Munkres()
        inds = m.compute(d)
        return inds
    
    def retrieve_ids(self, final_df:pd.DataFrame) -> pd.DataFrame:
        final_df = final_df.drop_duplicates(['dialogue'])
        final_df = self.dialogues.merge(final_df, on='dialogue', validate='many_to_one')
        final_df = final_df[['id', 'dialogue', 'final_summary']]
        return final_df