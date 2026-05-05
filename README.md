# Detecção de Discurso de Ódio (PT-BR)
# [Disponivel para visualização - StreamLit](https://plndiscursodeodio.streamlit.app/)

Trabalho da disciplina de Processamento de Linguagem Natural (5º semestre). Classificação multi-rótulo de discurso de ódio em português usando modelos clássicos (Naive Bayes, Regressão Logística, SVM Linear) sobre o dataset ToLD-BR, com app Streamlit para inferência interativa e armazenamento dos dados em **MongoDB Atlas**.

## Categorias

`homophobia`, `obscene`, `insult`, `racism`, `misogyny`, `xenophobia`

## Estrutura

```
Bases_de_dados/      # CSVs: HateBR, HateBRXplain, ToLD-BR
models/              # modelos treinados (.pkl) + metrics.json
db_manager.py        # conexão e operações com MongoDB Atlas
ingest.py            # ingestão dos dados CSV para o MongoDB (rodar uma vez)
PLN_Discurso_Odio.py # script de análise / experimentação
train_models.py      # treina NB, LR e SVM por categoria
app.py               # app Streamlit de inferência
```

## Pipeline

- Dados armazenados e carregados via **MongoDB Atlas** (banco não relacional).
- Pré-processamento com **spaCy** (`pt_core_news_sm`): lowercase, remoção de URLs/menções/RT, lematização, remoção de stopwords e pontuação.
- Vetorização **TF-IDF** (uni + bigramas, `max_features=12000`, `min_df=2`).
- Treino por categoria com `train_test_split` 70/30 estratificado.
- Modelos: `ComplementNB`, `LogisticRegression(class_weight='balanced')`, `LinearSVC(class_weight='balanced')`.
- Métricas salvas em `models/metrics.json` (accuracy, F1-macro, F1 da classe tóxica, matriz de confusão, classification report).


## Como rodar

### 1. Clonar o repositório
```bash
git clone https://github.com/nicolasaws1/pln-discurso-odio.git
cd pln-discurso-odio
```

### 2. Criar ambiente virtual e instalar dependências
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configurar o MongoDB Atlas
> ⚠️ **Obrigatório.** O projeto utiliza MongoDB Atlas como banco de dados não relacional para armazenamento e carregamento dos dados de treinamento.

1. Crie uma conta gratuita em [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
2. Crie um cluster **M0 Free**
3. Crie um usuário e copie a connection string
4. Na raiz do projeto, crie um arquivo `.env` baseado no `.env.example`:
```
MONGO_URI=mongodb+srv://<usuario>:<senha>@cluster0.xxxxx.mongodb.net/?appName=Cluster0
```

### 4. Ingerir os dados no MongoDB
> Rode **apenas uma vez** para popular o banco com os dados do CSV.
```bash
python ingest.py
```

### 5. Treinar os modelos
```bash
python train_models.py
```

### 6. Rodar o app
```bash
streamlit run app.py
```

# Ingestão dos dados no MongoDB (rodar apenas uma vez)
python ingest.py

# Treinar os modelos
python train_models.py

# App Streamlit
streamlit run app.py
```

## Dependências principais

`pandas`, `numpy`, `scikit-learn`, `spacy`, `joblib`, `streamlit`, `pymongo`, `python-dotenv`.

## Datasets

- **HateBR** / **HateBRXplain** — Vargas et al.
- **ToLD-BR** — Leite et al.

Os CSVs ficam em `Bases_de_dados/`. Use os datasets respeitando as licenças originais dos autores.
