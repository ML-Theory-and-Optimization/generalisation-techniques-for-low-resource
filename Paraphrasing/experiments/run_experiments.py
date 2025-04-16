import os
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.preprocessing import preprocess_text
from utils.embeddings import get_sentence_embeddings
from utils.classification import train_and_evaluate
from utils.vis import plot_accuracy_bar, plot_embedding_2D
from utils.paraphraser import paraphrase_tweet

LANGUAGE = "yoruba"  # Options: "yoruba", "igbo", "hausa", "pidgin"
EVAL_DIR = f"eval/{LANGUAGE}"
os.makedirs(EVAL_DIR, exist_ok=True)

TRAIN_PATH = "path to train data"
TEST_PATH = "path to test data"

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)


df_train = df_train[df_train['label'].isin(['negative', 'positive'])].copy()
df_test = df_test[df_test['label'].isin(['negative', 'positive'])].copy()
label_map = {'negative': 0, 'positive': 1}
df_train['label'] = df_train['label'].map(label_map).astype(int)
df_test['label'] = df_test['label'].map(label_map).astype(int)
print(f'shape of train set: {df_train.shape}')
print(f'shape of test set: {df_test.shape}')
print(f"Train set size: {len(df_train)}, Test set size: {len(df_test)}")
print(f"Train set unique labels: {df_train['label'].unique()}")
print(f"Test set unique labels: {df_test['label'].unique()}")
print(f'null values in train set: {df_train.isnull().sum()}')
print(f'null values in test set: {df_test.isnull().sum()}')
df_train.dropna( inplace=True, axis=0)
df_test.dropna(inplace=True, axis=0)
print(f'shape of train set after dropping null values: {df_train.shape}')
print(f'shape of test set after dropping null values: {df_test.shape}')



df_train['tweet'] = df_train['tweet'].apply(lambda x: preprocess_text(x, language=LANGUAGE))
df_train["paraphrased_tweet"] = df_train.apply(lambda row: paraphrase_tweet(row["tweet"], row["label"], language=LANGUAGE), axis=1)
df_test['tweet'] = df_test['tweet'].apply(lambda x: preprocess_text(x, language=LANGUAGE))


train_orig = df_train[['tweet', 'label']].copy()
train_para = df_train[['paraphrased_tweet', 'label']].copy()
train_combined = pd.concat([
    train_orig.sample(frac=0.5, random_state=42).rename(columns={'tweet': 'text'}),
    train_para.sample(frac=0.5, random_state=42).rename(columns={'paraphrased_tweet': 'text'})
])


train_orig.to_csv(f"{EVAL_DIR}/train_original.csv", index=False)
train_para.to_csv(f"{EVAL_DIR}/train_paraphrased.csv", index=False)
train_combined.to_csv(f"{EVAL_DIR}/train_combined.csv", index=False)


orig_train, orig_val = train_test_split(train_orig, test_size=0.2, stratify=train_orig['label'], random_state=42)
para_train, para_val = train_test_split(train_para, test_size=0.2, stratify=train_para['label'], random_state=42)
comb_train, comb_val = train_test_split(train_combined, test_size=0.2, stratify=train_combined['label'], random_state=42)


embedder = lambda x: get_sentence_embeddings(x.tolist())
Xo_train, Xo_val = embedder(orig_train['tweet']), embedder(orig_val['tweet'])
Xp_train, Xp_val = embedder(para_train['paraphrased_tweet']), embedder(para_val['paraphrased_tweet'])
Xc_train, Xc_val = embedder(comb_train['text']), embedder(comb_val['text'])


yo_train, yo_val = orig_train['label'], orig_val['label']
yp_train, yp_val = para_train['label'], para_val['label']
yc_train, yc_val = comb_train['label'], comb_val['label']


results = []
train_sets = [("Original", Xo_train, yo_train), ("Paraphrased", Xp_train, yp_train), ("Combined", Xc_train, yc_train)]
val_sets = [("Original", Xo_val, yo_val), ("Paraphrased", Xp_val, yp_val), ("Combined", Xc_val, yc_val)]

for t_name, X_tr, y_tr in train_sets:
    for v_name, X_va, y_va in val_sets:
        acc, _ = train_and_evaluate(X_tr, X_va, y_tr, y_va)
        results.append([LANGUAGE.title(), t_name, v_name, acc])


df_val_results = pd.DataFrame(results, columns=["Language", "Train Dataset", "Test Dataset", "Accuracy"])
plot_accuracy_bar(df_val_results, language=LANGUAGE.title(), save_path=f"{EVAL_DIR}/val_accuracy_plot.png")
df_val_results.to_csv(f"{EVAL_DIR}/val_results.csv", index=False)

plot_embedding_2D(Xo_val, yo_val.values, title=f"{LANGUAGE.title()} - Original (Val)", method='tsne', save_path=f"{EVAL_DIR}/val_embed_original_tsne.png")
plot_embedding_2D(Xp_val, yp_val.values, title=f"{LANGUAGE.title()} - Paraphrased (Val)", method='tsne', save_path=f"{EVAL_DIR}/val_embed_paraphrased_tsne.png")
plot_embedding_2D(Xc_val, yc_val.values, title=f"{LANGUAGE.title()} - Combined (Val)", method='tsne', save_path=f"{EVAL_DIR}/val_embed_combined_tsne.png")


# Final test evaluation
test_orig_embed = get_sentence_embeddings(df_test['tweet'].tolist())
test_labels = df_test['label']
final_results = []
for t_name, X_tr, y_tr in train_sets:
    acc, _ = train_and_evaluate(X_tr, test_orig_embed, y_tr, test_labels)
    final_results.append([LANGUAGE.title(), t_name, "Test Set", acc])

df_test_results = pd.DataFrame(final_results, columns=["Language", "Train Dataset", "Test Dataset", "Accuracy"])
plot_accuracy_bar(df_test_results, language=LANGUAGE.title(), save_path=f"{EVAL_DIR}/test_accuracy_plot.png")
df_test_results.to_csv(f"{EVAL_DIR}/test_results.csv", index=False)

plot_embedding_2D(test_orig_embed, test_labels.values, title=f"{LANGUAGE.title()} - Test Set", method='tsne', save_path=f"{EVAL_DIR}/test_embed_tsne.png")
