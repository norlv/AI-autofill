import logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pickle
import os
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__, template_folder='templates')

# Download NLTK Brown Corpus
try:
    import nltk
    nltk.download('brown', quiet=True)
except Exception as e:
    logging.error(f"Failed to download NLTK Brown Corpus: {e}")

# Load or cache SUBTLEX-US dataset
logging.info("Loading SUBTLEX-US dataset...")
subtlex_cache = 'data/subtlex_cache.pkl'
if os.path.exists(subtlex_cache):
    with open(subtlex_cache, 'rb') as f:
        df, common_words_list, word_freq_map, word_pos_map = pickle.load(f)
else:
    try:
        df = pd.read_excel('data/SUBTLEX-US.xlsx')
        df = df.dropna(subset=['Word', 'Zipf-value', 'Dom_PoS_SUBTLEX'])
    except FileNotFoundError:
        logging.warning("SUBTLEX-US.xlsx not found. Using fallback.")
        fallback_pairs = [
            ['hello', 'world'], ['machine', 'learning'], ['artificial', 'intelligence'], ['good', 'morning'],
            ['how', 'are'], ['data', 'science'], ['deep', 'learning'], ['new', 'york'], ['real', 'time'],
            ['natural', 'language'], ['coffee', 'shop'], ['text', 'generation'], ['chat', 'bot'], ['hello', 'there'],
            ['good', 'luck'], ['thank', 'you'], ['open', 'ai'], ['nice', 'day'], ['early', 'morning']
        ]
        df = pd.DataFrame({
            'Word': ['the', 'quick', 'machine', 'learning', 'hello', 'world', 'is', 'data', 'science', 'ai'] * 1000,
            'Zipf-value': [5.0, 4.5, 4.2, 4.0, 4.8, 4.3, 5.1, 4.4, 4.1, 4.0] * 1000,
            'Dom_PoS_SUBTLEX': ['det', 'adj', 'noun', 'noun', 'interj', 'noun', 'verb', 'noun', 'noun', 'noun'] * 1000
        })
    common_words_list = df['Word'].str.lower().tolist()
    word_freq_map = dict(zip(df['Word'].str.lower(), df['Zipf-value']))
    word_pos_map = dict(zip(df['Word'].str.lower(), df['Dom_PoS_SUBTLEX']))
    with open(subtlex_cache, 'wb') as f:
        pickle.dump((df, common_words_list, word_freq_map, word_pos_map), f)

# Load or cache embeddings
logging.info("Generating word embeddings...")
embedding_cache = 'data/embeddings_cache.pkl'
if os.path.exists(embedding_cache):
    with open(embedding_cache, 'rb') as f:
        word_embeddings, word_embedding_map = pickle.load(f)
else:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    word_embeddings = embedder.encode(common_words_list, show_progress_bar=True)
    word_embedding_map = dict(zip(common_words_list, word_embeddings))
    with open(embedding_cache, 'wb') as f:
        pickle.dump((word_embeddings, word_embedding_map), f)

# Prepare features
X_features = np.array([
    [
        word_freq_map.get(word, 0),
        1 if word_pos_map.get(word) in ['noun', 'verb', 'adjective'] else 0,
        *word_embeddings[i][:10]
    ] for i, word in enumerate(common_words_list)
])

# Load word-pair dataset and generate negative pairs
logging.info("Loading word-pair dataset...")
try:
    word_pair_df = pd.read_csv('data/word_pairs.csv')
    logging.info(f"Loaded {len(word_pair_df)} word pairs")
except FileNotFoundError:
    logging.error("word_pairs.csv not found. Using expanded fallback.")
    fallback_pairs = [
        ['hello', 'world'], ['machine', 'learning'], ['artificial', 'intelligence'], ['good', 'morning'],
        ['how', 'are'], ['data', 'science'], ['deep', 'learning'], ['new', 'york'], ['real', 'time'],
        ['natural', 'language'], ['coffee', 'shop'], ['text', 'generation'], ['chat', 'bot'], ['hello', 'there'],
        ['good', 'luck'], ['thank', 'you'], ['open', 'ai'], ['nice', 'day'], ['early', 'morning']
    ]
    word_pair_df = pd.DataFrame(fallback_pairs, columns=['word1', 'word2'])

# Generate negative pairs
negative_pairs = []
existing_pairs = set((row['word1'], row['word2']) for _, row in word_pair_df.iterrows())
for _ in range(len(word_pair_df)):
    word1 = random.choice(common_words_list)
    word2 = random.choice(common_words_list)
    while (word1, word2) in existing_pairs:
        word1 = random.choice(common_words_list)
        word2 = random.choice(common_words_list)
    negative_pairs.append([word1, word2])

# Combine positive and negative pairs
all_pairs_df = pd.concat([
    word_pair_df.assign(label=1),
    pd.DataFrame(negative_pairs, columns=['word1', 'word2']).assign(label=0)
], axis=0)

X_pair_features = np.array([
    np.concatenate([
        word_embedding_map.get(row['word1'], np.zeros(384)),
        word_embedding_map.get(row['word2'], np.zeros(384))
    ]) for _, row in all_pairs_df.iterrows()
])
y_pair_labels = all_pairs_df['label'].values

# Initialize models
logging.info("Training machine learning models...")
models = {}
try:
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Fix: set pad token to eos token
    transformer_model = AutoModelForCausalLM.from_pretrained('distilgpt2')
except Exception as e:
    logging.error(f"Failed to load DistilGPT2: {e}")
    tokenizer = None
    transformer_model = None

try:
    rf_pair = RandomForestClassifier(n_estimators=100, max_depth=10)
    if len(X_pair_features) > 0 and len(np.unique(y_pair_labels)) > 1:
        rf_pair.fit(X_pair_features, y_pair_labels)
        models['random_forest_pair'] = rf_pair
    else:
        logging.warning("Insufficient pair data for random_forest_pair")
        models['random_forest_pair'] = None

    gb_pair = GradientBoostingClassifier(n_estimators=100, max_depth=3)
    if len(X_pair_features) > 0 and len(np.unique(y_pair_labels)) > 1:
        gb_pair.fit(X_pair_features, y_pair_labels)
        models['gradient_boosting_pair'] = gb_pair
    else:
        logging.warning("Insufficient pair data for gradient_boosting_pair")
        models['gradient_boosting_pair'] = None

    transactions = [row.tolist() for _, row in word_pair_df.iterrows()]
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df_transactions, min_support=0.002, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    models['apriori'] = rules
    logging.info(f"Apriori rules available: {len(rules)}")

except Exception as e:
    logging.error(f"Error training models: {e}")


def algorithm_suggestion(text, method):
    logging.info(f"Processing input: '{text}', Method: {method}")
    ends_with_space = text.endswith(' ')  # Check raw input for trailing space
    text_stripped = text.strip()          # Strip for splitting into words
    words = text_stripped.split()

    if ends_with_space:
        # User finished a word, so next word prediction mode
        prefix = ''  # no prefix for autocomplete since word finished
        context_words = words[-2:] if len(words) >= 2 else words[-1:]
    else:
        # User typing current word, autocomplete mode
        prefix = words[-1].lower() if words else ''
        context_words = words[:-1][-2:] if len(words) > 1 else []

    logging.info(f"Ends with space: {ends_with_space}, Context words: {context_words}, Prefix: '{prefix}'")

    # Default fallback suggestions: autocomplete by prefix or semantic similarity for next word
    if ends_with_space and context_words:
        # Next-word mode: find semantic similarity with context words
        context_embeddings = [
            word_embedding_map.get(w.lower(), np.zeros(384)) for w in context_words
        ]
        context_embedding = np.mean(context_embeddings, axis=0) if context_embeddings else np.zeros(384)
        similarities = [
            (word, cosine_similarity([context_embedding], [word_embedding_map[word]])[0][0])
            for word in common_words_list
        ]
        default_suggestions = [word for word, _ in sorted(similarities, key=lambda x: x[1], reverse=True)][:5]
    else:
        # Autocomplete mode
        default_suggestions = sorted(
            [word for word in common_words_list if not prefix or word.startswith(prefix)],
            key=lambda x: word_freq_map.get(x, 0),
            reverse=True
        )[:5]

    # --- Now select method ---

    if method == 'apriori':
        rules = models.get('apriori')
        suggestions = []
        if ends_with_space and context_words:
            context_set = set(w.lower() for w in context_words)
            for _, rule in rules.iterrows():
                antecedents = set(list(rule['antecedents']))
                if antecedents.issubset(context_set) or context_set & antecedents:
                    suggestions.extend([w for w in list(rule['consequents']) if w in common_words_list])
            suggestions = sorted(list(set(suggestions)), key=lambda x: word_freq_map.get(x, 0), reverse=True)[:5]
            logging.info(f"Apriori suggestions based on context {context_set}: {suggestions}")
            return suggestions if suggestions else default_suggestions
        else:
            suggestions = [
                word for word in common_words_list if not prefix or word.startswith(prefix)
            ]
            return sorted(suggestions, key=lambda x: word_freq_map.get(x, 0), reverse=True)[:5] or default_suggestions


    elif method in ['random_forest_pair', 'gradient_boosting_pair']:
        model = models.get(method)
        if model is None:
            logging.error(f"Model {method} not initialized")
            return default_suggestions
        if ends_with_space and context_words:
            context_embeddings = [
                word_embedding_map.get(w.lower(), np.zeros(384)) for w in context_words
            ]
            context_embedding = np.mean(context_embeddings, axis=0) if context_embeddings else np.zeros(384)
            input_features = np.array([
                np.concatenate([
                    context_embedding,
                    word_embedding_map.get(word, np.zeros(384))
                ]) for word in common_words_list
            ])
            try:
                scores = model.predict_proba(input_features)[:, 1]
                suggestions = [word for _, word in sorted(zip(scores, common_words_list), reverse=True)][:5]
                logging.info(f"{method} suggestions based on context {context_words}: {suggestions}")
                return suggestions if suggestions else default_suggestions
            except Exception as e:
                logging.error(f"{method} error: {e}")
                return default_suggestions
        else:
            suggestions = [word for word in common_words_list if not prefix or word.startswith(prefix)]
            return sorted(suggestions, key=lambda x: word_freq_map.get(x, 0), reverse=True)[:5] or default_suggestions

    elif method == 'SUBTLEX':
        suggestions = [word for word in common_words_list if not prefix or word.startswith(prefix)]
        return sorted(suggestions, key=lambda x: word_freq_map.get(x, 0), reverse=True)[:5] or default_suggestions

    elif method == 'transformer':
        if tokenizer is None or transformer_model is None:
            logging.error("Transformer model not initialized")
            return default_suggestions
        try:
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            outputs = transformer_model.generate(
                **inputs,
                max_new_tokens=1,
                num_return_sequences=3,
                pad_token_id=tokenizer.eos_token_id
            )
            suggestions = [
                tokenizer.decode(output, skip_special_tokens=True)[len(text):].strip()
                for output in outputs
            ]
            suggestions = [
                s for s in suggestions if s in common_words_list and (not prefix or s.startswith(prefix))
            ][:5]
            logging.info(f"Transformer suggestions: {suggestions}")
            return suggestions if suggestions else default_suggestions
        except Exception as e:
            logging.error(f"Transformer error: {e}")
            return default_suggestions

    return default_suggestions


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/suggest', methods=['POST'])
def suggest():
    data = request.get_json()
    text = data.get('text', '')
    method = data.get('method', 'SUBTLEX')
    try:
        suggestions = algorithm_suggestion(text, method)
        logging.info(f"Returning suggestions for '{text}': {suggestions}")
        return jsonify({'suggestions': suggestions})
    except Exception as e:
        logging.error(f"Error generating suggestions: {e}")
        return jsonify({'suggestions': [], 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
