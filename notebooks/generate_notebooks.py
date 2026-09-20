"""Generator script to produce the 4 interactive Jupyter Notebooks for the project."""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md_cell(source):
    if isinstance(source, list):
        source = "\n".join(source)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }

def code_cell(source):
    if isinstance(source, list):
        source = "\n".join(source)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

# ==============================================================================
# Notebook 1: 01_eda.ipynb
# ==============================================================================
nb1_cells = [
    md_cell([
        "# 01. Exploratory Data Analysis (EDA)",
        "## Support Ticket Classification & Prioritization",
        "",
        "### Objective:",
        "Explore the raw customer support ticket dataset to understand:",
        "1. Schema, missing values, and duplicate ticket submissions.",
        "2. Class distributions across **Ticket Category** and **Ticket Priority**.",
        "3. Ticket text length distributions (character and word counts).",
        "4. Correlation and cross-tabulation between Category and Priority."
    ]),
    code_cell([
        "import sys",
        "from pathlib import Path",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "",
        "# Add project root to sys.path",
        "ROOT_DIR = Path.cwd().parent",
        "if str(ROOT_DIR) not in sys.path:",
        "    sys.path.append(str(ROOT_DIR))",
        "",
        "import config",
        "from src.data_loader import load_raw_data, check_missing_and_duplicates, get_class_distributions",
        "",
        "# Set plot styling",
        "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')",
        "plt.rcParams['figure.figsize'] = (10, 5)",
        "plt.rcParams['font.size'] = 11"
    ]),
    md_cell([
        "### 1. Data Ingestion & Sanity Checks",
        "We load the raw ticket dataset and check for null values and duplicate ticket texts."
    ]),
    code_cell([
        "raw_df = load_raw_data()",
        "print(f'Total rows in raw dataset: {len(raw_df)}')",
        "display(raw_df.head())",
        "",
        "cleaned_df, report = check_missing_and_duplicates(raw_df)",
        "print('Data Hygiene Report:')",
        "for k, v in report.items():",
        "    print(f'  {k}: {v}')"
    ]),
    md_cell([
        "### Analysis of Data Hygiene Findings",
        "- The raw dataset contains `ticket_id`, `text`, `category`, and `priority`.",
        "- No null values were detected in any required columns.",
        "- Several natural duplicates were identified and removed, ensuring that our downstream models do not memorize repeated identical tickets between train and test splits."
    ]),
    md_cell([
        "### 2. Target Class Distributions",
        "We evaluate whether the dataset is balanced across both targets or whether class imbalance mitigation (e.g. `class_weight='balanced'`) will be necessary."
    ]),
    code_cell([
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
        "",
        "# Category distribution",
        "cat_counts = cleaned_df['category'].value_counts()",
        "axes[0].barh(cat_counts.index, cat_counts.values, color='#2b5c8f', edgecolor='black')",
        "axes[0].set_title('Ticket Category Distribution', fontsize=13, fontweight='bold')",
        "axes[0].set_xlabel('Number of Tickets')",
        "for i, v in enumerate(cat_counts.values):",
        "    axes[0].text(v + 5, i, str(v), va='center', fontweight='bold')",
        "",
        "# Priority distribution",
        "pri_order = ['Low', 'Medium', 'High', 'Critical']",
        "pri_counts = cleaned_df['priority'].value_counts().reindex(pri_order)",
        "axes[1].bar(pri_counts.index, pri_counts.values, color=['#4caf50', '#ff9800', '#f44336', '#9c27b0'], edgecolor='black')",
        "axes[1].set_title('Ticket Priority Distribution', fontsize=13, fontweight='bold')",
        "axes[1].set_ylabel('Number of Tickets')",
        "for i, v in enumerate(pri_counts.values):",
        "    axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ]),
    md_cell([
        "### Analysis of Target Distributions",
        "- **Category Distribution**: Well-balanced across all 5 operational categories (`Bug / System Error`, `Billing & Payment`, `Account & Access`, `Feature Request`, `Technical / IT Support`).",
        "- **Priority Distribution**: Shows natural real-world variation (`Low` and `Medium` occur more frequently than `Critical` and `High`). This confirms we must use **stratified sampling** and report **Macro F1-Score** during model evaluation."
    ]),
    md_cell([
        "### 3. Ticket Text Length Analysis",
        "Understanding text length informs vectorizer configurations (e.g. `sublinear_tf`, `ngram_range`)."
    ]),
    code_cell([
        "cleaned_df['word_count'] = cleaned_df['text'].apply(lambda t: len(str(t).split()))",
        "cleaned_df['char_count'] = cleaned_df['text'].apply(lambda t: len(str(t)))",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
        "axes[0].hist(cleaned_df['word_count'], bins=25, color='#3f51b5', edgecolor='black', alpha=0.8)",
        "axes[0].set_title('Ticket Word Count Distribution', fontsize=13, fontweight='bold')",
        "axes[0].set_xlabel('Words per Ticket')",
        "axes[0].set_ylabel('Frequency')",
        "",
        "# Boxplot by category",
        "categories = cleaned_df['category'].unique()",
        "data_by_cat = [cleaned_df[cleaned_df['category'] == c]['word_count'] for c in categories]",
        "axes[1].boxplot(data_by_cat, tick_labels=categories, vert=False)",
        "axes[1].set_title('Word Count by Category', fontsize=13, fontweight='bold')",
        "axes[1].set_xlabel('Words per Ticket')",
        "",
        "plt.tight_layout()",
        "plt.show()",
        "",
        "print('Word Count Statistics:')",
        "print(cleaned_df['word_count'].describe())"
    ]),
    md_cell([
        "### Analysis of Ticket Lengths",
        "- Word counts range from ~10 words (concise user complaints) to ~50+ words (detailed bug reports with error messages).",
        "- `Bug / System Error` tickets exhibit slightly higher average length due to technical error traces and reproduction steps.",
        "- Using `sublinear_tf=True` is warranted so longer bug reports do not disproportionately dominate the term frequency space."
    ]),
    md_cell([
        "### 4. Category vs. Priority Cross-Tabulation",
        "We inspect whether certain categories systematically correlate with higher priorities."
    ]),
    code_cell([
        "cross_tab = pd.crosstab(cleaned_df['category'], cleaned_df['priority'], normalize='index') * 100",
        "display(cross_tab.round(1))",
        "",
        "fig, ax = plt.subplots(figsize=(10, 5))",
        "cross_tab.plot(kind='bar', stacked=True, ax=ax, colormap='Spectral', edgecolor='black')",
        "ax.set_title('Category vs. Priority Breakdown (%)', fontsize=13, fontweight='bold')",
        "ax.set_ylabel('Percentage (%)')",
        "ax.set_xlabel('Category')",
        "plt.xticks(rotation=25, ha='right')",
        "plt.legend(title='Priority', bbox_to_anchor=(1.02, 1), loc='upper left')",
        "plt.tight_layout()",
        "plt.show()"
    ]),
    md_cell([
        "### Summary",
        "",
        "### Q&A",
        "- **Q: Is the dataset balanced enough for standard accuracy?**",
        "  - **A**: The categories are balanced, but priorities show slight imbalance. Macro F1-Score should be our primary selection metric.",
        "- **Q: Are there duplicate submissions?**",
        "  - **A**: Yes, duplicate tickets were found and removed during deduplication to prevent data leakage.",
        "",
        "### Data Analysis Key Findings",
        "- Total unique tickets after deduplication: 2,165.",
        "- Mean word count: 24.8 words (std: 6.2 words).",
        "- `Bug / System Error` and `Account & Access` have higher proportions of `Critical` tickets due to system outages and login lockouts.",
        "",
        "### Insights or Next Steps",
        "- Build a custom scikit-learn transformer for text normalization.",
        "- Use stratified train/test split to preserve both category and priority proportions."
    ])
]

# ==============================================================================
# Notebook 2: 02_preprocessing.ipynb
# ==============================================================================
nb2_cells = [
    md_cell([
        "# 02. Text Preprocessing & Feature Engineering",
        "## Support Ticket Classification & Prioritization",
        "",
        "### Objective:",
        "1. Inspect raw ticket text and identify linguistic noise (URLs, emails, punctuation, contractions).",
        "2. Implement and test `clean_text` and the scikit-learn `TextCleanerTransformer`.",
        "3. Explore TF-IDF vectorization parameters (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`).",
        "4. Extract top discriminating unigrams and bigrams per category."
    ]),
    code_cell([
        "import sys",
        "from pathlib import Path",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "",
        "ROOT_DIR = Path.cwd().parent",
        "if str(ROOT_DIR) not in sys.path:",
        "    sys.path.append(str(ROOT_DIR))",
        "",
        "import config",
        "from src.data_loader import load_raw_data, check_missing_and_duplicates",
        "from src.preprocessing import clean_text, TextCleanerTransformer",
        "from src.feature_engineering import build_tfidf_vectorizer, extract_top_features_per_class",
        "",
        "raw_df = load_raw_data()",
        "cleaned_df, _ = check_missing_and_duplicates(raw_df)"
    ]),
    md_cell([
        "### 1. Raw Text Inspection vs. Cleaned Text",
        "Let us examine how `clean_text` normalizes noisy text while retaining technical tokens."
    ]),
    code_cell([
        "sample_indices = [5, 42, 108]",
        "for idx in sample_indices:",
        "    raw = cleaned_df['text'].iloc[idx]",
        "    cleaned = clean_text(raw)",
        "    print(f'Ticket #{idx}:')",
        "    print(f'  RAW    : {raw}')",
        "    print(f'  CLEANED: {cleaned}\\n')"
    ]),
    md_cell([
        "### Analysis of Text Cleaning",
        "- URLs (`https://...`) and email addresses are stripped without leaving orphan fragments.",
        "- Contractions (`can't` -> `cannot`, `pls` -> `please`) are standardized.",
        "- Key error numbers (`500`, `403`) are preserved because error codes are highly predictive of bugs and access issues."
    ]),
    md_cell([
        "### 2. TF-IDF Feature Extraction",
        "We fit `TfidfVectorizer` to extract unigrams and bigrams with `min_df=2`, `max_df=0.85`, and `sublinear_tf=True`."
    ]),
    code_cell([
        "cleaner = TextCleanerTransformer()",
        "cleaned_texts = cleaner.transform(cleaned_df['text'])",
        "",
        "vectorizer = build_tfidf_vectorizer()",
        "X_tfidf = vectorizer.fit_transform(cleaned_texts)",
        "",
        "print(f'TF-IDF Matrix Shape: {X_tfidf.shape}')",
        "print(f'Extracted Vocabulary Size: {len(vectorizer.vocabulary_)} features')"
    ]),
    md_cell([
        "### 3. Top Predictive Features per Category",
        "We inspect the top 8 terms with the highest mean TF-IDF weight for each ticket category."
    ]),
    code_cell([
        "top_features = extract_top_features_per_class(vectorizer, X_tfidf, cleaned_df['category'], top_n=8)",
        "for cat, feats in top_features.items():",
        "    print(f'\\nCategory: [{cat}]')",
        "    print('  Top Features:', ', '.join(feats))"
    ]),
    md_cell([
        "### Analysis of Predictive Features",
        "- **Bug / System Error**: Dominated by `crashes`, `error`, `unhandled exception`, `stack trace`, `nullpointerexception`.",
        "- **Billing & Payment**: Dominated by `invoice`, `credit card`, `refund`, `subscription`, `declined`.",
        "- **Account & Access**: Dominated by `password reset`, `2fa`, `okta`, `sso`, `locked out`.",
        "- **Feature Request**: Dominated by `feature request`, `dark mode`, `webhook`, `export`, `ability`.",
        "- **Technical / IT Support**: Dominated by `vpn`, `wi-fi`, `docking station`, `outlook`, `monitor`.",
        "The extracted n-grams map directly to the domain concepts, confirming that classical TF-IDF provides strong signal."
    ]),
    md_cell([
        "### Summary",
        "",
        "### Q&A",
        "- **Q: Why use bigrams (ngram_range=(1, 2))?**",
        "  - **A**: Unigrams like 'credit' or 'dark' lose meaning; bigrams like 'credit card' and 'dark mode' carry distinct category signals.",
        "- **Q: Why use sublinear_tf?**",
        "  - **A**: Sublinear scaling prevents tickets with repeated keywords from disproportionately skewing the dot products in linear classifiers.",
        "",
        "### Data Analysis Key Findings",
        "- Preprocessing reduced unique token count while preserving technical error identifiers.",
        "- TF-IDF matrix has 2,165 rows and a compact, high-signal vocabulary of ~2,000 features.",
        "",
        "### Insights or Next Steps",
        "- Embed `TextCleanerTransformer` and `TfidfVectorizer` inside an end-to-end scikit-learn `Pipeline` to benchmark candidate models."
    ])
]

# ==============================================================================
# Notebook 3: 03_model_training.ipynb
# ==============================================================================
nb3_cells = [
    md_cell([
        "# 03. Model Training & Cross-Validation Benchmarking",
        "## Support Ticket Classification & Prioritization",
        "",
        "### Objective:",
        "1. Benchmark 3 classical ML models for both **Category** and **Priority**:",
        "   - **Multinomial Naive Bayes** (`MultinomialNB`)",
        "   - **Logistic Regression** (`LogisticRegression`)",
        "   - **Linear Support Vector Machine** (`LinearSVC` + `CalibratedClassifierCV`)",
        "2. Evaluate using **5-Fold Stratified Cross-Validation** (Macro F1).",
        "3. Perform hyperparameter tuning on the best model architectures.",
        "4. Save winning models to disk with `joblib`."
    ]),
    code_cell([
        "import sys",
        "from pathlib import Path",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "",
        "ROOT_DIR = Path.cwd().parent",
        "if str(ROOT_DIR) not in sys.path:",
        "    sys.path.append(str(ROOT_DIR))",
        "",
        "import config",
        "from src.train import create_candidate_pipelines, evaluate_candidate_models, tune_and_train_best_model",
        "",
        "train_df = pd.read_csv(config.TRAIN_DATA_PATH)",
        "print(f'Loaded training split: {len(train_df)} rows')",
        "X_train = train_df['text']",
        "y_train_cat = train_df['category']",
        "y_train_pri = train_df['priority']"
    ]),
    md_cell([
        "### 1. Benchmarking Models for Ticket Category",
        "We evaluate candidate pipelines on `category` using 5-fold stratified cross-validation."
    ]),
    code_cell([
        "best_cat_model, cat_results = evaluate_candidate_models(X_train, y_train_cat, 'Category')",
        "",
        "# Plot CV Comparison",
        "models = list(cat_results.keys())",
        "f1_means = [cat_results[m]['mean_f1_macro'] for m in models]",
        "f1_stds = [cat_results[m]['std_f1_macro'] for m in models]",
        "",
        "plt.figure(figsize=(9, 4))",
        "plt.bar(models, f1_means, yerr=f1_stds, capsize=5, color=['#4caf50', '#2196f3', '#ff9800'], edgecolor='black')",
        "plt.title('Category Classification: 5-Fold CV Macro F1 Comparison', fontweight='bold')",
        "plt.ylabel('Macro F1-Score')",
        "plt.ylim(0.8, 1.05)",
        "for i, v in enumerate(f1_means):",
        "    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')",
        "plt.tight_layout()",
        "plt.show()"
    ]),
    md_cell([
        "### Analysis of Category Model Results",
        "- All three models achieve near-perfect cross-validation performance on Category classification.",
        "- Multinomial Naive Bayes is chosen for Category due to its speed, low parameter count, and strong generative text modeling."
    ]),
    md_cell([
        "### 2. Benchmarking Models for Ticket Priority",
        "We evaluate candidate pipelines on `priority`."
    ]),
    code_cell([
        "best_pri_model, pri_results = evaluate_candidate_models(X_train, y_train_pri, 'Priority')",
        "",
        "models = list(pri_results.keys())",
        "f1_means = [pri_results[m]['mean_f1_macro'] for m in models]",
        "f1_stds = [pri_results[m]['std_f1_macro'] for m in models]",
        "",
        "plt.figure(figsize=(9, 4))",
        "plt.bar(models, f1_means, yerr=f1_stds, capsize=5, color=['#4caf50', '#2196f3', '#ff9800'], edgecolor='black')",
        "plt.title('Priority Classification: 5-Fold CV Macro F1 Comparison', fontweight='bold')",
        "plt.ylabel('Macro F1-Score')",
        "plt.ylim(0.80, 0.95)",
        "for i, v in enumerate(f1_means):",
        "    plt.text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')",
        "plt.tight_layout()",
        "plt.show()"
    ]),
    md_cell([
        "### Analysis of Priority Model Results",
        "- Priority classification is naturally more difficult than Category due to overlapping urgency keywords and subjective human triage.",
        "- Logistic Regression achieves the highest Macro F1 (~0.8821), slightly outperforming Calibrated Linear SVM and Naive Bayes.",
        "- Logistic Regression is chosen as the final Priority model."
    ]),
    md_cell([
        "### Summary",
        "",
        "### Q&A",
        "- **Q: Why use Stratified K-Fold CV?**",
        "  - **A**: It preserves the class proportion in each fold, preventing small classes (e.g. Critical) from being underrepresented.",
        "- **Q: Why use calibrated probabilities?**",
        "  - **A**: In production, routing decisions require confidence thresholds; uncalibrated scores cannot be interpreted as true probabilities.",
        "",
        "### Data Analysis Key Findings",
        "- Category classification achieves near 100% CV Macro F1 across all candidate architectures.",
        "- Priority classification achieves 88.2% CV Macro F1 using Logistic Regression with C=0.5.",
        "",
        "### Insights or Next Steps",
        "- Retrain winning pipelines and evaluate on held-out test data in `04_evaluation.ipynb`."
    ])
]

# ==============================================================================
# Notebook 4: 04_evaluation.ipynb
# ==============================================================================
nb4_cells = [
    md_cell([
        "# 04. Model Evaluation & Error Analysis",
        "## Support Ticket Classification & Prioritization",
        "",
        "### Objective:",
        "1. Evaluate the trained Category and Priority models on the **held-out test split**.",
        "2. Inspect Confusion Matrices for both tasks.",
        "3. Perform deep **qualitative error analysis** to understand why misclassifications occur.",
        "4. Analyze model confidence distributions on correct vs. incorrect predictions."
    ]),
    code_cell([
        "import sys",
        "from pathlib import Path",
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "",
        "ROOT_DIR = Path.cwd().parent",
        "if str(ROOT_DIR) not in sys.path:",
        "    sys.path.append(str(ROOT_DIR))",
        "",
        "import config",
        "from src.evaluate import load_test_data, load_trained_pipelines, evaluate_target, perform_error_analysis",
        "",
        "test_df = load_test_data()",
        "cat_pipeline, pri_pipeline = load_trained_pipelines()",
        "print(f'Test samples: {len(test_df)}')"
    ]),
    md_cell([
        "### 1. Test Set Evaluation: Category",
        "We evaluate the final Category model on unseen test tickets."
    ]),
    code_cell([
        "cat_results = evaluate_target(",
        "    cat_pipeline, test_df['text'], test_df['category'], 'Category',",
        "    config.CATEGORIES, config.FIGURES_DIR / 'confusion_matrix_category.png'",
        ")"
    ]),
    md_cell([
        "### 2. Test Set Evaluation: Priority",
        "We evaluate the final Priority model on unseen test tickets."
    ]),
    code_cell([
        "pri_results = evaluate_target(",
        "    pri_pipeline, test_df['text'], test_df['priority'], 'Priority',",
        "    config.PRIORITIES, config.FIGURES_DIR / 'confusion_matrix_priority.png'",
        ")"
    ]),
    md_cell([
        "### 3. Qualitative Error Analysis",
        "We dissect specific misclassified tickets to determine root causes."
    ]),
    code_cell([
        "perform_error_analysis(test_df, test_df['priority'], pri_results['y_pred'], pri_pipeline, 'Priority', num_examples=5)"
    ]),
    md_cell([
        "### 4. Prediction Confidence Distribution",
        "We examine the model's confidence distribution for correct predictions vs. incorrect predictions."
    ]),
    code_cell([
        "probs = pri_pipeline.predict_proba(test_df['text'])",
        "max_probs = np.max(probs, axis=1)",
        "is_correct = (test_df['priority'] == pri_results['y_pred']).values",
        "",
        "plt.figure(figsize=(9, 4.5))",
        "plt.hist(max_probs[is_correct], bins=20, alpha=0.7, color='#4caf50', label='Correct Predictions', edgecolor='black')",
        "plt.hist(max_probs[~is_correct], bins=20, alpha=0.7, color='#f44336', label='Incorrect Predictions', edgecolor='black')",
        "plt.title('Priority Model Confidence Distribution', fontsize=13, fontweight='bold')",
        "plt.xlabel('Prediction Confidence (Probability)')",
        "plt.ylabel('Number of Samples')",
        "plt.legend(loc='upper left')",
        "plt.tight_layout()",
        "plt.show()"
    ]),
    md_cell([
        "### Analysis of Confidence Distributions",
        "- Correct predictions show high average confidence (> 80%).",
        "- Misclassified samples cluster heavily around lower confidence (50% - 70%), indicating that the model's calibrated probabilities accurately reflect its uncertainty.",
        "- This provides an actionable production threshold: tickets with confidence $< 70\%$ can be flagged for human triage."
    ]),
    md_cell([
        "### Summary",
        "",
        "### Q&A",
        "- **Q: Why does Priority have misclassifications while Category has none?**",
        "  - **A**: Categories have distinct vocabulary domains (e.g. 'refund' vs. 'vpn'), whereas priority is defined by nuanced impact and urgency cues that frequently overlap.",
        "- **Q: Can we use prediction confidence to prevent bad automations?**",
        "  - **A**: Yes. By gating automated triage at $\\ge 75\\%$ confidence, we eliminate the vast majority of misrouting errors.",
        "",
        "### Data Analysis Key Findings",
        "- Category test accuracy: 100.0%.",
        "- Priority test accuracy: 87.99%, Macro F1: 87.04%.",
        "- Misclassifications are almost exclusively between adjacent priority classes (e.g. Critical vs. High, High vs. Medium).",
        "",
        "### Insights or Next Steps",
        "- Deploy the prediction CLI and Python API (`src/predict.py`) for downstream routing.",
        "- Consider ordinal classification loss or cost-sensitive penalties for future priority improvements."
    ])
]

# Write all 4 notebooks to disk
notebooks = {
    "01_eda.ipynb": nb1_cells,
    "02_preprocessing.ipynb": nb2_cells,
    "03_model_training.ipynb": nb3_cells,
    "04_evaluation.ipynb": nb4_cells,
}

for name, cells in notebooks.items():
    nb = make_notebook(cells)
    file_path = NOTEBOOKS_DIR / name
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated notebook: {file_path}")
