from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util
class Symentic_word_manager:
    def __init__(self):
        # Load a pre-trained Sentence-Transformer self.model
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Function to generate potential unigram candidates using WordNet
    def generate_candidates(self,bigram):
        words = bigram.split()
        candidates = set()

        # Handle negations like "not" etc
        if words[0].lower() in ["not","never","none","neither","nor","without","lacking"
                                                    "missing","absent","fail to"]:
            primary_word = words[1]  # Focus on the second word
            # Add antonyms for the negated word
            for syn in wordnet.synsets(primary_word):
                for lemma in syn.lemmas():
                    if lemma.antonyms():
                        candidates.add(lemma.antonyms()[0].name().replace("_", " "))
            # Add negation-based transformations (e.g., "incomplete" from "complete")
            candidates.add("in" + primary_word)
            candidates.add("un" + primary_word)

        # Add synonyms for the bigram or the primary word
        bigram_wordnet_key = bigram.replace(" ", "_")  # Handle bigrams as phrases
        for syn in wordnet.synsets(bigram_wordnet_key):
            for lemma in syn.lemmas():
                candidates.add(lemma.name().replace("_", " "))

        return list(candidates)

    # Function to find the best matching unigram for a bigram
    def bigram_to_unigram(self,bigram):
        # Generate unigram candidates dynamically
        unigram_candidates = self.generate_candidates(bigram)

        # If no candidates are found, return the input bigram as a fallback
        if not unigram_candidates:
            return bigram

        # Encode the bigram and unigram candidates
        bigram_embedding = self.model.encode(bigram, convert_to_tensor=True)
        unigram_embeddings = self.model.encode(unigram_candidates, convert_to_tensor=True)

        # Compute cosine similarity
        similarities = util.pytorch_cos_sim(bigram_embedding, unigram_embeddings)

        # Find the unigram with the highest similarity
        best_match_index = similarities.argmax().item()
        return unigram_candidates[best_match_index]

