import nltk

# Download the required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Read the input text file
with open('input.txt', 'r') as f:
    text = f.read()

# Tokenize the text into sentences and words
sentences = nltk.sent_tokenize(text)
words = [word.lower() for sentence in sentences for word in nltk.word_tokenize(sentence)]

# Tag the words with part-of-speech (POS) labels
tagged_words = nltk.pos_tag(words)

# Extract difficult and important words based on their POS tags
difficult_words = set([word for word, tag in tagged_words if tag.startswith('J') or tag.startswith('V')])
important_words = set([word for word, tag in tagged_words if tag.startswith('N') or tag.startswith('J')])

# Write the vocabulary list to a file
with open('vocabulary.txt', 'w') as f:
    f.write('Difficult words:\n')
    f.write('\n'.join(sorted(difficult_words)))
    f.write('\n\n')
    f.write('Important words:\n')
    f.write('\n'.join(sorted(important_words)))
