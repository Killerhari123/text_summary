from transformers.utils import hub
import nltk
import heapq
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
from transformers import T5ForConditionalGeneration, T5Tokenizer


def extractive_summarization(text, num_sentences=3):
    sentences = sent_tokenize(text)
    word_frequencies = defaultdict(int)
    
    for word in word_tokenize(text):
        word_frequencies[word.lower()] += 1
    
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] /= max_frequency
    
    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]
    
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    return ' '.join(summary_sentences)

def abstractive_summarization(text):
    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    summary_ids = model.generate(input_ids, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    text = """Text summarization is the process of shortening a text while preserving key information. There are two main types: extractive summarization, which selects important sentences from the text, and abstractive summarization, which generates new sentences to summarize the main idea. Various machine learning models like T5 and BART are commonly used for abstractive summarization."""
    
    print("Extractive Summary:")
    print(extractive_summarization(text))
    
    print("\nAbstractive Summary:")
    print(abstractive_summarization(text))
