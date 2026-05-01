import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import time

# Baixar stopwords em português do nltk
nltk.download('stopwords')
stop_words_pt = stopwords.words('portuguese')

BASE_DIR = Path(__file__).resolve().parent

# Função para carregar a base de dados
def carregar_arquivo():
    caminho = BASE_DIR / "Bases_de_dados" / "ToLD-BR.csv"
    try:
        df = pd.read_csv(caminho)
        st.write(df.head())  # Exibe as primeiras linhas do arquivo
        return df
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado em: {caminho}")
        return None

# Função para treinar os modelos para cada categoria
def treinar_modelo(df, categoria):
    # Pré-processamento dos dados
    df['text'] = df['text'].apply(lambda x: x.lower())
    df['text'] = df['text'].apply(lambda x: x.strip())

    tfidf = TfidfVectorizer(stop_words=stop_words_pt, max_features=5000)  # Usando stopwords em português
    X = tfidf.fit_transform(df['text'])
    y = df[categoria]  # Utiliza a coluna correspondente à categoria (ex: 'homophobia')

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinando os modelos
    modelos = {
        'Naive Bayes': MultinomialNB(),
        'Regressão Logística': LogisticRegression(),
        'SVM Linear': LinearSVC(max_iter=2000)
    }

    resultados = {}
    tempos_resposta = {}  # Dicionário para armazenar o tempo de resposta

    for nome, modelo in modelos.items():
        # Iniciar o temporizador
        start_time = time.time()

        # Treinamento do modelo
        modelo.fit(X_train, y_train)

        # Fim do temporizador para treinamento
        treino_time = time.time() - start_time

        # Predição
        start_time = time.time()
        y_pred = modelo.predict(X_test)
        pred_time = time.time() - start_time

        # Calcular a acurácia
        acc = accuracy_score(y_test, y_pred)

        # Armazenar os resultados
        resultados[nome] = acc
        tempos_resposta[nome] = {'treinamento': treino_time, 'predicao': pred_time}

    return tfidf, resultados, tempos_resposta

# Função principal do Streamlit
def main():
    st.title("Análise de Discurso de Ódio")

    # Carregar a base de dados
    df = carregar_arquivo()

    if df is not None:
        # Campo para inserir uma frase
        texto_novo = st.text_area("Digite uma frase para análise:")

        if st.button("Classificar"):
            if texto_novo:
                # Inicializar resultados de classificação
                resultados_classificacao = {}
                acuracia_modelos = {}
                tempos_modelos = {}

                # Categorias de discurso de ódio a verificar
                categorias = ['homophobia', 'obscene', 'insult', 'racism', 'misogyny', 'xenophobia']

                # Armazenar resultados em listas para a tabela
                tabela_classificacao = []
                tabela_acuracia = []
                tabela_tempos = []

                # Treinar e classificar para cada categoria
                for categoria in categorias:
                    # Treinamento do modelo para cada categoria
                    tfidf, resultados, tempos_resposta = treinar_modelo(df, categoria)
                    
                    # Classificar a nova frase
                    texto_vectorizado = tfidf.transform([texto_novo])

                    predicoes = {}
                    for modelo, acuracia in resultados.items():
                        if modelo == 'Naive Bayes':
                            modelo_treinado = MultinomialNB().fit(tfidf.transform(df['text']), df[categoria])
                            predicoes[modelo] = modelo_treinado.predict(texto_vectorizado)[0]
                        elif modelo == 'Regressão Logística':
                            modelo_treinado = LogisticRegression().fit(tfidf.transform(df['text']), df[categoria])
                            predicoes[modelo] = modelo_treinado.predict(texto_vectorizado)[0]
                        elif modelo == 'SVM Linear':
                            modelo_treinado = LinearSVC(max_iter=2000).fit(tfidf.transform(df['text']), df[categoria])
                            predicoes[modelo] = modelo_treinado.predict(texto_vectorizado)[0]
                    
                    # Armazenar as predições e as acurácias
                    resultados_classificacao[categoria] = predicoes
                    acuracia_modelos[categoria] = resultados
                    tempos_modelos[categoria] = tempos_resposta

                    # Preencher as tabelas de classificação, acurácia e tempos
                    row_classificacao = {'Categoria': categoria.capitalize()}
                    row_acuracia = {'Categoria': categoria.capitalize()}
                    row_tempos = {'Categoria': categoria.capitalize()}

                    for modelo, classe in predicoes.items():
                        row_classificacao[modelo] = classe
                    for modelo, acuracia in resultados.items():
                        row_acuracia[modelo] = f'{acuracia * 100:.2f}%'
                    for modelo, tempos in tempos_resposta.items():
                        row_tempos[modelo] = f'Treinamento: {tempos["treinamento"]:.4f} s, Predição: {tempos["predicao"]:.4f} s'

                    tabela_classificacao.append(row_classificacao)
                    tabela_acuracia.append(row_acuracia)
                    tabela_tempos.append(row_tempos)

                # Exibir resultados de classificação em formato de tabela
                st.write("Classificação do texto por categoria e modelo:")
                df_classificacao = pd.DataFrame(tabela_classificacao)
                st.table(df_classificacao)

                # Exibir resultados de acurácia dos modelos
                st.write("Acurácia dos modelos para cada categoria:")
                df_acuracia = pd.DataFrame(tabela_acuracia)
                st.table(df_acuracia)

                # Exibir tempos de resposta dos modelos
                st.write("Tempo de resposta de cada modelo (em segundos):")
                df_tempos = pd.DataFrame(tabela_tempos)
                st.table(df_tempos)

    else:
        st.warning("A base de dados não foi carregada corretamente.")

if __name__ == "__main__":
    main()