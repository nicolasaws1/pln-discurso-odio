# PLN — Detecção de Discurso de Ódio (PT-BR)

Trabalho da disciplina de Processamento de Linguagem Natural (5º semestre). Classificação multi-rótulo de discurso de ódio em português usando modelos clássicos (Naive Bayes, Regressão Logística, SVM Linear) sobre os datasets HateBR, HateBRXplain e ToLD-BR, com app Streamlit para inferência interativa.

## Categorias

`homophobia`, `obscene`, `insult`, `racism`, `misogyny`, `xenophobia`

## Estrutura

```
Bases_de_dados/      # CSVs: HateBR, HateBRXplain, ToLD-BR
models/              # modelos treinados (.pkl) + metrics.json
PLN_Discurso_Odio.py # script de análise / experimentação
train_models.py      # treina NB, LR e SVM por categoria
app.py               # app Streamlit de inferência
```

## Pipeline

- Pré-processamento com **spaCy** (`pt_core_news_sm`): lowercase, remoção de URLs/menções/RT, lematização, remoção de stopwords e pontuação.
- Vetorização **TF-IDF** (uni + bigramas, `max_features=12000`, `min_df=2`).
- Treino por categoria com `train_test_split` 70/30 estratificado.
- Modelos: `ComplementNB`, `LogisticRegression(class_weight='balanced')`, `LinearSVC(class_weight='balanced')`.
- Métricas salvas em `models/metrics.json` (accuracy, F1-macro, F1 da classe tóxica, matriz de confusão, classification report).

## Como rodar

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download pt_core_news_sm

# Treinar (opcional — modelos já vêm no repo)
python train_models.py

# App Streamlit
streamlit run app.py
```

## Dependências principais

`pandas`, `numpy`, `scikit-learn`, `spacy`, `joblib`, `streamlit`.

## Datasets

- **HateBR** / **HateBRXplain** — Vargas et al.
- **ToLD-BR** — Leite et al.

Os CSVs ficam em `Bases_de_dados/`. Use os datasets respeitando as licenças originais dos autores.
