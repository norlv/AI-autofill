import flask
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.decomposition import PCA
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

data_path = os.path.join('data', 'SUBTLEX-US.xlsx')
try:
    subtlex_df = pd.read_excel(data_path)
    word_col = next((col for col in ['Word', 'word', 'WORD'] if col in subtlex_df.columns), None)
    freq_col = next((col for col in ['Zipf', 'Zipf-value', 'zipf', 'Lg10WF', 'SUBTLWF'] if col in subtlex_df.columns),
                    None)
    pos_col = next((col for col in ['Dom_PoS_SUBTLEX', 'PoS'] if col in subtlex_df.columns), None)
    if not word_col or not freq_col:
        raise ValueError(f"Could not find word or frequency columns. Found: {list(subtlex_df.columns)}")
    freq_threshold = 3.0
    common_words = subtlex_df[subtlex_df[freq_col] > freq_threshold][[word_col, freq_col, pos_col]].dropna(
        subset=[word_col, freq_col])
    common_words_list = common_words[word_col].str.lower().tolist()
    word_freq_map = dict(zip(common_words[word_col].str.lower(), common_words[freq_col]))
    word_pos_map = dict(zip(common_words[word_col].str.lower(), common_words[pos_col])) if pos_col else {}
    logger.info(
        f"Loaded {len(common_words_list)} common words from SUBTLEX-US dataset using columns '{word_col}' and '{freq_col}'")
    if pos_col:
        logger.info(f"Unique PoS values: {common_words[pos_col].unique()}")
except Exception as e:
    logger.error(f"Failed to load SUBTLEX-US dataset: {e}")

# Sentence transformer for embeddings
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    word_embeddings = embedder.encode(common_words_list)
    word_embedding_map = dict(zip(common_words_list, word_embeddings))
    logger.info("Loaded sentence transformer embeddings")
except Exception as e:
    logger.error(f"Failed to load sentence transformer: {e}")
    word_embeddings = np.random.rand(len(common_words_list), 384)
    word_embedding_map = dict(zip(common_words_list, word_embeddings))

# Synthetic word-pair dataset for Apriori
synthetic_pairs = pd.DataFrame([
    ['hello', 'world'], ['world', 'is'], ['is', 'beautiful'], ['happy', 'day'], ['help', 'me'],
    ['hello', 'there'], ['world', 'peace'], ['is', 'awesome'], ['happy', 'life'], ['help', 'now']
], columns=['word1', 'word2'])
synthetic_transactions = pd.DataFrame(np.zeros((len(synthetic_pairs), len(common_words_list)), dtype=int),
                                      columns=common_words_list)
for idx, row in synthetic_pairs.iterrows():
    if row['word1'] in common_words_list:
        synthetic_transactions.at[idx, row['word1']] = 1
    if row['word2'] in common_words_list:
        synthetic_transactions.at[idx, row['word2']] = 1

# Features for classification/regression
X_features = np.array([[
    word_freq_map[word],
    1 if isinstance(word_pos_map.get(word), str) and word_pos_map.get(word).lower() in ['noun', 'verb',
                                                                                        'adjective'] else 0,
    *word_embedding_map[word][:10]  # First 10 embedding dimensions
] for word in common_words_list])
y_labels = np.array(
    [1 if word_freq_map[word] > 4 else 0 for word in common_words_list])  # High-frequency words as positive

# Pre-train algorithms
models = {}
try:
    lr = LinearRegression()
    lr.fit(X_features, list(word_freq_map.values()))
    models['linear_regression'] = lr

    log_reg = LogisticRegression()
    log_reg.fit(X_features, y_labels)
    models['logistic_regression'] = log_reg

    dt = DecisionTreeClassifier()
    dt.fit(X_features, y_labels)
    models['decision_tree'] = dt

    rf = RandomForestClassifier()
    rf.fit(X_features, y_labels)
    models['random_forest'] = rf

    nb = GaussianNB()
    nb.fit(X_features, y_labels)
    models['naive_bayes'] = nb

    knn = KNeighborsClassifier()
    knn.fit(X_features, y_labels)
    models['knn'] = knn

    svm = SVC(probability=True)
    svm.fit(X_features, y_labels)
    models['svm'] = svm

    gb = GradientBoostingClassifier()
    gb.fit(X_features, y_labels)
    models['gradient_boosting'] = gb

    kmeans = KMeans(n_clusters=10)
    kmeans.fit(word_embeddings)
    models['kmeans'] = kmeans
    word_clusters = kmeans.predict(word_embeddings)
    word_cluster_map = dict(zip(common_words_list, word_clusters))

    frequent_itemsets = apriori(synthetic_transactions, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    models['apriori'] = rules

    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(word_embeddings)
    models['pca'] = pca
    word_pca_map = dict(zip(common_words_list, X_pca))

    logger.info("Pre-trained all machine learning models")
except Exception as e:
    logger.error(f"Failed to pre-train models: {e}")

# Load DistilGPT2 for text generation
try:
    generator = pipeline('text-generation', model='distilgpt2', framework='tf')
    logger.info("DistilGPT2 model loaded successfully with TensorFlow")
except Exception as e:
    logger.error(f"Failed to load DistilGPT2 model: {e}")
    generator = None


# Simple autocomplete using SUBTLEX-US common words
def simple_autocomplete(prefix, max_suggestions=5):
    try:
        suggestions = [word for word in common_words_list if word.startswith(prefix.lower())]
        suggestions = sorted(suggestions, key=lambda x: word_freq_map.get(x, 0), reverse=True)
        return suggestions[:max_suggestions]
    except Exception as e:
        logger.error(f"Simple autocomplete failed: {e}")
        return []


# Transformer-based suggestion
def transformer_suggestion(text, max_length=10, num_sequences=5):
    if not generator:
        logger.warning("Transformer model is not loaded")
        return ""
    try:
        results = generator(text, max_length=max_length, num_return_sequences=num_sequences, return_full_text=False)
        suggestions = [result['generated_text'].strip().split()[-1] for result in results if
                       result['generated_text'].strip()]
        valid_suggestions = [s for s in suggestions if s.lower() in common_words_list or len(s) < 10]
        return valid_suggestions[0] if valid_suggestions else suggestions[0] if suggestions else ""
    except Exception as e:
        logger.error(f"Transformer suggestion failed: {e}")
        return ""


# Algorithm-based suggestion
def algorithm_suggestion(text, method, max_suggestions=5):
    try:
        prefix = text.split()[-1] if text.split() else text
        is_complete_word = text.strip().endswith(' ')
        suggestions = []

        if method == 'subtlex':
            return simple_autocomplete(prefix, max_suggestions)
        elif method == 'transformer':
            return [transformer_suggestion(text, max_suggestions=max_suggestions)]

        # Classification/Regression Algorithms
        if method in ['linear_regression', 'logistic_regression', 'decision_tree', 'random_forest',
                      'naive_bayes', 'knn', 'svm', 'gradient_boosting']:
            model = models.get(method)
            if not model:
                return []
            if is_complete_word:
                # For next-word, use context (last word's embedding)
                last_word = text.split()[-2] if len(text.split()) > 1 else ''
                if last_word.lower() in word_embedding_map:
                    input_features = np.array([[
                        word_freq_map.get(word, 0),
                        1 if isinstance(word_pos_map.get(word), str) and word_pos_map.get(word).lower() in ['noun',
                                                                                                            'verb',
                                                                                                            'adjective'] else 0,
                        *word_embedding_map[word][:10]
                    ] for word in common_words_list])
                    if method == 'linear_regression':
                        scores = model.predict(input_features)
                    else:
                        scores = model.predict_proba(input_features)[:, 1]
                    suggestions = [word for _, word in sorted(zip(scores, common_words_list), reverse=True)]
                else:
                    suggestions = [word for word in common_words_list]
            else:
                suggestions = [word for word in common_words_list if word.startswith(prefix.lower())]

        # K-Means Clustering
        elif method == 'kmeans':
            if is_complete_word:
                last_word = text.split()[-2] if len(text.split()) > 1 else ''
                if last_word.lower() in word_cluster_map:
                    cluster = word_cluster_map[last_word.lower()]
                    suggestions = [word for word in common_words_list if word_cluster_map.get(word) == cluster]
                else:
                    suggestions = common_words_list
            else:
                suggestions = [word for word in common_words_list if word.startswith(prefix.lower())]

        # Apriori
        elif method == 'apriori':
            rules = models.get('apriori', pd.DataFrame())
            if is_complete_word:
                last_word = text.split()[-2].lower() if len(text.split()) > 1 else ''
                suggestions = []
                for _, rule in rules.iterrows():
                    antecedents = list(rule['antecedents'])
                    consequents = list(rule['consequents'])
                    if last_word in antecedents:
                        suggestions.extend([w for w in consequents if w in common_words_list])
                suggestions = list(set(suggestions)) or common_words_list
            else:
                suggestions = [word for word in common_words_list if word.startswith(prefix.lower())]

        # PCA
        elif method == 'pca':
            if is_complete_word:
                last_word = text.split()[-2] if len(text.split()) > 1 else ''
                if last_word.lower() in word_pca_map:
                    last_pca = word_pca_map[last_word.lower()]
                    distances = [np.linalg.norm(word_pca_map[word] - last_pca) for word in common_words_list]
                    suggestions = [word for _, word in sorted(zip(distances, common_words_list))]
                else:
                    suggestions = common_words_list
            else:
                suggestions = [word for word in common_words_list if word.startswith(prefix.lower())]

        # SIFT (simulated with frequency ranking)
        elif method == 'sift':
            suggestions = sorted(common_words_list, key=lambda x: word_freq_map.get(x, 0), reverse=True)
            if not is_complete_word:
                suggestions = [word for word in suggestions if word.startswith(prefix.lower())]

        return suggestions[:max_suggestions]
    except Exception as e:
        logger.error(f"Algorithm suggestion failed for {method}: {e}")
        return []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/suggest', methods=['POST'])
def suggest():
    try:
        data = request.json
        text = data.get('text', '')
        method = data.get('method', 'subtlex')  # Default to subtlex
        if not text:
            logger.info("Empty input received")
            return jsonify({'suggestions': []})

        suggestions = algorithm_suggestion(text, method)
        logger.info(f"Suggestions generated for input: {text} (method: {method}) -> {suggestions}")
        return jsonify({'suggestions': suggestions[:5]})
    except Exception as e:
        logger.error(f"Suggestion endpoint failed: {e}")
        return jsonify({'suggestions': []}), 500


if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to start Flask server: {e}")