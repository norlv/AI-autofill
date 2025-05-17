import pandas as pd
import logging
import nltk
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# Download NLTK Brown Corpus
nltk.download('brown', quiet=True)

# Default word pairs (curated subset)
default_pairs = [
    ['the', 'quick'], ['quick', 'brown'], ['brown', 'fox'], ['fox', 'jumped'],
    ['jumped', 'over'], ['over', 'the'], ['the', 'lazy'], ['lazy', 'dog'],
    ['machine', 'learning'], ['learning', 'is'], ['is', 'fun'], ['fun', 'to'],
    ['to', 'explore'], ['explore', 'new'], ['new', 'ideas'], ['ideas', 'for'],
    ['for', 'ai'], ['ai', 'projects'], ['projects', 'are'], ['are', 'exciting'],
    ['data', 'science'], ['science', 'is'], ['is', 'growing'], ['growing', 'fast'],
    ['fast', 'and'], ['and', 'efficient'], ['efficient', 'algorithms'], ['algorithms', 'drive'],
    ['drive', 'innovation'], ['hello', 'world'], ['world', 'is'], ['is', 'full'],
    ['full', 'of'], ['of', 'opportunities'], ['opportunities', 'for'], ['for', 'students'],
    ['students', 'to'], ['to', 'learn'], ['learn', 'and'], ['and', 'grow'],
    ['artificial', 'intelligence'], ['intelligence', 'can'], ['can', 'transform'],
    ['transform', 'industries'], ['industries', 'like'], ['like', 'healthcare'],
    ['healthcare', 'and'], ['and', 'finance'], ['this', 'is'], ['is', 'a']
]

# Load SUBTLEX-US for vocabulary filtering
logging.info("Loading SUBTLEX-US dataset...")
try:
    df = pd.read_excel('data/SUBTLEX-US.xlsx')
    df = df.dropna(subset=['Word', 'Zipf-value'])
    common_words = set(df['Word'].str.lower().tolist())
    zipf_values = dict(zip(df['Word'].str.lower(), df['Zipf-value']))
except FileNotFoundError:
    logging.warning("SUBTLEX-US.xlsx not found. Using fallback vocabulary.")
    common_words = set([pair[0] for pair in default_pairs] + [pair[1] for pair in default_pairs])
    zipf_values = {w: 4.0 for w in common_words}

# Generate additional pairs from Brown Corpus
def generate_brown_pairs(max_pairs=950):
    word_pairs = []
    for sentence in nltk.corpus.brown.sents()[:5000]:
        for i in range(len(sentence) - 1):
            word1 = sentence[i].lower()
            word2 = sentence[i + 1].lower()
            if (word1 in common_words and word2 in common_words and
                zipf_values.get(word1, 0) > 4.0 and zipf_values.get(word2, 0) > 4.0):
                word_pairs.append([word1, word2])
            if len(word_pairs) >= max_pairs:
                break
        if len(word_pairs) >= max_pairs:
            break
    return word_pairs

# Save to CSV
def save_word_pairs(word_pairs):
    df_pairs = pd.DataFrame(word_pairs, columns=['word1', 'word2'])
    df_pairs.drop_duplicates(inplace=True)
    df_pairs.to_csv('data/word_pairs.csv', index=False)
    logging.info(f"Saved word_pairs.csv with {len(df_pairs)} pairs")

# Main execution
if __name__ == "__main__":
    # Check for existing curated pairs
    curated_pairs = []
    try:
        df_curated = pd.read_csv('data/word_pairs.csv')
        if not df_curated.empty and all(col in df_curated.columns for col in ['word1', 'word2']):
            curated_pairs = df_curated[['word1', 'word2']].values.tolist()
            logging.info(f"Loaded {len(curated_pairs)} curated pairs")
        else:
            logging.warning("word_pairs.csv is empty or invalid. Using default pairs.")
            curated_pairs = default_pairs
    except (FileNotFoundError, pd.errors.EmptyDataError):
        logging.warning("word_pairs.csv not found or empty. Using default pairs.")
        curated_pairs = default_pairs

    # Add Brown Corpus pairs to reach ~1,000
    brown_pairs = generate_brown_pairs(max_pairs=1000 - len(curated_pairs))
    word_pairs = curated_pairs + brown_pairs

    # Remove duplicates and limit to 1,000 pairs
    word_pairs = list(dict.fromkeys(map(tuple, word_pairs)))[:1000]

    # Save
    save_word_pairs(word_pairs)