import os
import time
import feedparser
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import quote
import praw  
from googleapiclient.discovery import build  
from LeIA import SentimentIntensityAnalyzer  
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta


# Configuração visual dos gráficos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# -------------------------------------------------------------------
# CONFIGURAÇÕES E CHAVES DE API
# -------------------------------------------------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

analyzer = SentimentIntensityAnalyzer()


import os

# O código vai buscar essas chaves nas variáveis de ambiente do GitHub
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")

KEYWORDS_NOTICIAS = ["vacina", "vacinação", "antivax", "movimento antivacina", "imunização"]
KEYWORDS_DEBATE = ["vacina covid", "antivax brasil", "vacinação obrigatória", "efeito colateral vacina"]

# -------------------------------------------------------------------
# MÓDULOS DE EXTRAÇÃO DE DADOS
# -------------------------------------------------------------------
def extrair_noticias_rss(termos, limite_por_termo=250):
    noticias = []
    print("🌐 [1/3] Coletando notícias de portais (Google News)...")
    
    for termo in termos:
        termo_encoded = quote(termo)
        rss_url = f"https://news.google.com/rss/search?q={termo_encoded}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
        feed = feedparser.parse(rss_url)
        
        for entry in feed.entries[:limite_por_termo]:
            url = entry.link
            titulo = entry.title
            data_publicacao = entry.published
            
            texto = ""
            try:
                resp = requests.get(url, headers=HEADERS, timeout=8)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    paragrafos = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 40]
                    texto = " ".join(paragrafos)
            except Exception:
                texto = entry.get("summary", "")

            noticias.append({
                "fonte_tipo": "Notícia Portal",
                "termo_busca": termo,
                "titulo": titulo,
                "texto": texto if texto else titulo,
                "data": data_publicacao,
                "url": url
            })
            time.sleep(0.3)
            
    print(f"✅ {len(noticias)} notícias coletadas.")
    return noticias

def extrair_comentarios_youtube(termos, api_key, max_videos_por_termo=3, max_comentarios_por_video=250):
    comentarios_yt = []
    if not api_key:
        print("⚠️ Chave da API do YouTube não configurada. Pulando coleta do YouTube.")
        return comentarios_yt

    print("🎥 [2/3] Coletando comentários de vídeos do YouTube...")
    try:
        youtube = build("youtube", "v3", developerKey=api_key)

        for termo in termos:
            search_response = youtube.search().list(
                q=termo,
                part="id,snippet",
                maxResults=max_videos_por_termo,
                type="video",
                relevanceLanguage="pt"
            ).execute()

            for video_item in search_response.get("items", []):
                video_id = video_item["id"]["videoId"]
                video_titulo = video_item["snippet"]["title"]

                try:
                    comment_response = youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        maxResults=max_comentarios_por_video,
                        textFormat="plainText"
                    ).execute()

                    for comment_item in comment_response.get("items", []):
                        comment_data = comment_item["snippet"]["topLevelComment"]["snippet"]
                        texto_comentario = comment_data["textDisplay"].strip()

                        if len(texto_comentario) > 10:
                            comentarios_yt.append({
                                "fonte_tipo": "Comentário YouTube",
                                "termo_busca": termo,
                                "titulo": f"Vídeo: {video_titulo}",
                                "texto": texto_comentario,
                                "data": comment_data["publishedAt"],
                                "url": f"https://www.youtube.com/watch?v={video_id}"
                            })
                except Exception:
                    continue
    except Exception as e:
        print(f"❌ Erro ao conectar com a API do YouTube: {e}")

    print(f"✅ {len(comentarios_yt)} comentários do YouTube coletados.")
    return comentarios_yt

def extrair_comentarios_reddit(termos, client_id=None, client_secret=None, limite=3):
    comentarios = []
    if not client_id or not client_secret:
        print("⚠️ Credenciais do Reddit não fornecidas. Pulando coleta do Reddit.")
        return comentarios

    print("💬 [3/3] Coletando posts e comentários do Reddit...")
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="VacinaScraperScript/1.0"
        )

        for termo in termos:
            for submission in reddit.subreddit("brasil+brasilivre+conversas").search(termo, limit=limite):
                comentarios.append({
                    "fonte_tipo": "Post Fórum",
                    "termo_busca": termo,
                    "titulo": submission.title,
                    "texto": submission.selftext if submission.selftext else submission.title,
                    "data": pd.to_datetime(submission.created_utc, unit='s').strftime('%Y-%m-%d %H:%M:%S'),
                    "url": f"https://reddit.com{submission.permalink}"
                })
                
                submission.comments.replace_more(limit=0)
                for comment in submission.comments[:10]:
                    if len(comment.body.strip()) > 15:
                        comentarios.append({
                            "fonte_tipo": "Comentário Fórum",
                            "termo_busca": termo,
                            "titulo": f"Re: {submission.title[:50]}...",
                            "texto": comment.body,
                            "data": pd.to_datetime(comment.created_utc, unit='s').strftime('%Y-%m-%d %H:%M:%S'),
                            "url": f"https://reddit.com{comment.permalink}"
                        })
    except Exception as e:
        print(f"❌ Erro na coleta do Reddit: {e}")

    print(f"✅ {len(comentarios)} posts/comentários do Reddit coletados.")
    return comentarios

# -------------------------------------------------------------------
# PROCESSAMENTO E NLP (LEIA-BR)
# -------------------------------------------------------------------
def classificar_sentimento_e_rotulo(texto):
    scores = analyzer.polarity_scores(texto)
    compound = scores["compound"]
    
    texto_lower = texto.lower()
    termos_criticos = [
        "vachina", "experimento", "chips", "efetividade zero", 
        "forçar vacina", "ditadura sanitária", "efeito colateral",
        "miocardite", "clorocina", "trombo", "placebo"
    ]
    
    eh_antivax_provavel = any(t in texto_lower for t in termos_criticos)
    
    if compound >= 0.05:
        categoria = "Favorável / Positivo"
    elif compound <= -0.05:
        categoria = "Discurso Crítico / Antivax Prob" if eh_antivax_provavel else "Crítico / Negativo"
    else:
        categoria = "Neutro / Informativo"
        
    return compound, categoria

# -------------------------------------------------------------------
# GERAÇÃO DA PÁGINA HTML COM FILTRO DE ANO
# -------------------------------------------------------------------
def gerar_dashboard_html(df):
    print("\n🌐 Gerando Dashboard HTML...")
    
    template_html = """
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Observatório de Saúde: Vacinas</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background-color: #f4f7f6; color: #333; line-height: 1.6; }
            header { background-color: #ffffff; border-bottom: 3px solid #0056b3; padding: 1.5rem 5%; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .logo h1 { margin: 0; color: #0056b3; font-size: 2rem; }
            .container { width: 95%; max-width: 1400px; margin: 2rem auto; }
            
            .section-title { font-size: 1.8rem; color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 0.5rem; margin-bottom: 1.5rem; display: inline-block; width: 100%; }
            
            /* Gráficos */
            .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 4rem; }
            .chart-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; }
            .chart-card img { max-width: 100%; height: auto; border-radius: 4px; }
            
            /* Caixas de Top 10 */
            .top-lists-wrapper { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 4rem; }
            .top-list-box { flex: 1; min-width: 350px; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
            .top-list-box h3 { margin-top: 0; border-bottom: 2px solid #eee; padding-bottom: 15px; font-size: 1.5rem; }
            .title-pos { color: #2b9348; border-bottom-color: #2b9348 !important; }
            .title-neg { color: #d90429; border-bottom-color: #d90429 !important; }
            
            .top-list { list-style: none; padding: 0; margin: 0; }
            .top-list li { border-bottom: 1px solid #eee; padding: 15px 0; }
            .top-list li:last-child { border-bottom: none; }
            .top-list-item h4 { margin: 5px 0; font-size: 1.1rem; color: #222; }
            .top-list-item p { font-size: 0.9rem; color: #555; margin: 10px 0; }
            .date-stamp { font-size: 0.75rem; color: #888; display: block; margin-top: 5px; }
            
            /* Filtro de Ano */
            .filter-container { background: white; padding: 15px 25px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; }
            .filter-container select { padding: 10px; font-size: 1rem; border: 1px solid #ccc; border-radius: 5px; min-width: 150px; cursor: pointer; }
            .filter-container label { font-weight: bold; font-size: 1.1rem; color: #0056b3; }

            /* Feed Geral */
            .news-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .news-card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); display: flex; flex-direction: column; justify-content: space-between; transition: all 0.3s; }
            .news-card h3 { margin: 0 0 10px 0; font-size: 1.2rem; color: #222; }
            .news-card p { font-size: 0.95rem; color: #666; margin-bottom: 15px; flex-grow: 1; }
            .news-footer { display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #eee; padding-top: 10px; }
            
            /* Utilitários */
            .read-more { color: #0056b3; text-decoration: none; font-weight: bold; font-size: 0.9rem; }
            .read-more:hover { text-decoration: underline; }
            .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; color: white; text-transform: uppercase; }
            .badge-positivo { background-color: #2b9348; }
            .badge-negativo { background-color: #d90429; }
            .badge-neutro { background-color: #8d99ae; }
            .badge-fonte { background-color: #333; margin-right: 5px;}
            
            footer { background-color: #222; color: white; text-align: center; padding: 2rem 0; margin-top: 3rem; }
        </style>
    </head>
    <body>
        <header>
            <div class="logo">
                <h1>🔬 Observatório de Saúde: Vacinas</h1>
            </div>
            <div style="color: #666; font-weight: bold;">Monitoramento Analítico e NLP</div>
        </header>

        <div class="container">
            <!-- Gráficos Gerados -->
            <h2 class="section-title">Análise de Dados e Sentimentos</h2>
            <div class="charts-grid">
                <div class="chart-card">
                    <h3>Sentimentos por Fonte</h3>
                    <img src="grafico_sentimento_por_fonte.png" alt="Gráfico Sentimentos por Fonte">
                </div>
                <div class="chart-card">
                    <h3>Evolução Temporal</h3>
                    <img src="grafico_evolucao_temporal.png" alt="Gráfico Evolução Temporal">
                </div>
                <div class="chart-card">
                    <h3>Distribuição Geral</h3>
                    <img src="grafico_distribuicao_sentimentos.png" alt="Gráfico Distribuição Geral">
                </div>
            </div>

            <!-- Caixas de TOP 10 (Últimos 7 dias) -->
            <h2 class="section-title">Destaques da Semana (Últimos 7 Dias)</h2>
            <div class="top-lists-wrapper">
                <div class="top-list-box">
                    <h3 class="title-pos">Top 10 Textos Mais Positivos</h3>
                    <ul class="top-list">
                        [LISTA_POSITIVAS]
                    </ul>
                </div>
                <div class="top-list-box">
                    <h3 class="title-neg">Top 10 Textos Mais Negativos / Críticos</h3>
                    <ul class="top-list">
                        [LISTA_NEGATIVAS]
                    </ul>
                </div>
            </div>

            <!-- Header do Feed com Filtro -->
            <h2 class="section-title" style="border:none; margin-bottom: 0;">Amostra do Feed Geral (Histórico)</h2>
            
            <div class="filter-container">
                <label for="filtroAno">📅 Filtrar publicações pelo ano:</label>
                <select id="filtroAno">
                    <option value="todos">Todos os Anos</option>
                    [OPCOES_ANOS]
                </select>
            </div>

            <div class="news-grid" id="containerFeed">
                [CARDS_NOTICIAS]
            </div>
        </div>

        <footer>
            <p>&copy; 2026 Observatório de Saúde gerado via Python Scraping.</p>
        </footer>

        <!-- Script JavaScript para fazer o filtro funcionar dinamicamente -->
        <script>
            document.getElementById('filtroAno').addEventListener('change', function() {
                var anoSelecionado = this.value;
                var cartoes = document.querySelectorAll('.news-card');

                cartoes.forEach(function(cartao) {
                    var anoCartao = cartao.getAttribute('data-ano');
                    
                    if (anoSelecionado === 'todos' || anoCartao === anoSelecionado) {
                        cartao.style.display = 'flex'; // Mostra o cartão
                    } else {
                        cartao.style.display = 'none'; // Oculta o cartão
                    }
                });
            });
        </script>
    </body>
    </html>
    """

    # --- 1. CRIAR AS OPÇÕES DO DROPDOWN (ANOS) ---
    # Pegamos os anos únicos que não são nulos, transformamos em inteiros, e ordenamos do mais novo para o mais velho
    anos_unicos = sorted([int(a) for a in df['ano'].dropna().unique()], reverse=True)
    opcoes_anos_html = "".join([f'<option value="{ano}">{ano}</option>' for ano in anos_unicos])
    template_html = template_html.replace("[OPCOES_ANOS]", opcoes_anos_html)

    # --- 2. FILTRAR OS DADOS DOS ÚLTIMOS 7 DIAS (PARA OS TOP 10) ---
    hoje = pd.Timestamp.utcnow()
    limite_7_dias = hoje - pd.Timedelta(days=7)
    df_7_dias = df[df["data_dt"] >= limite_7_dias]

    top_positivas = df_7_dias.sort_values(by="score_sentimento", ascending=False).head(10)
    top_negativas = df_7_dias.sort_values(by="score_sentimento", ascending=True).head(10)

    def formatar_lista(df_subset, badge_color_class):
        if df_subset.empty:
            return "<li><div class='top-list-item'><p><i>Nenhuma publicação encontrada neste recorte de 7 dias.</i></p></div></li>"
            
        html = ""
        for _, row in df_subset.iterrows():
            titulo = str(row['titulo'])[:70] + "..." if len(str(row['titulo'])) > 70 else str(row['titulo'])
            texto = str(row['texto'])[:120] + "..." if len(str(row['texto'])) > 120 else str(row['texto'])
            data_str = str(row['data_dt'])[:10]
            
            html += f"""
            <li>
                <div class="top-list-item">
                    <span class="badge {badge_color_class}">Score: {row['score_sentimento']:.2f}</span>
                    <span class="badge badge-fonte">{row['fonte_tipo']}</span>
                    <h4>{titulo}</h4>
                    <p>{texto}</p>
                    <a href="{row['url']}" target="_blank" class="read-more">Ler na íntegra &rarr;</a>
                    <span class="date-stamp">Publicado em: {data_str}</span>
                </div>
            </li>
            """
        return html

    html_positivas = formatar_lista(top_positivas, "badge-positivo")
    html_negativas = formatar_lista(top_negativas, "badge-negativo")

    template_html = template_html.replace("[LISTA_POSITIVAS]", html_positivas)
    template_html = template_html.replace("[LISTA_NEGATIVAS]", html_negativas)

    # --- 3. GERAR FEED GERAL COM AS ETIQUETAS DE ANO (DATA-ANO) ---
    df_feed = df.head(100) # Aumentado para 100, para ter o que filtrar ao mudar o ano
    cards_html = ""
    for _, row in df_feed.iterrows():
        badge_class = "badge-neutro"
        if "Positivo" in row['classificacao_preliminar']:
            badge_class = "badge-positivo"
        elif "Crítico" in row['classificacao_preliminar'] or "Antivax" in row['classificacao_preliminar']:
            badge_class = "badge-negativo"

        texto_resumo = str(row['texto'])[:180] + "..." if len(str(row['texto'])) > 180 else str(row['texto'])
        
        # Define o ano do cartão (se houver problema na data, define como desconhecido)
        ano_cartao = str(int(row['ano'])) if pd.notnull(row['ano']) else 'desconhecido'

        # Repare na adição do `data-ano="{ano_cartao}"` na div abaixo. O JavaScript usa isso!
        card = f"""
        <div class="news-card" data-ano="{ano_cartao}">
            <div>
                <span class="badge badge-fonte">{row['fonte_tipo']}</span>
                <span class="badge {badge_class}">{row['classificacao_preliminar']}</span>
                <h3>{row['titulo'][:60]}...</h3>
                <p>{texto_resumo}</p>
            </div>
            <div class="news-footer">
                <span style="font-size: 0.8rem; color: #999;">{str(row['data_dt'])[:10]}</span>
                <a href="{row['url']}" target="_blank" class="read-more">Ver Fonte &rarr;</a>
            </div>
        </div>
        """
        cards_html += card

    html_final = template_html.replace("[CARDS_NOTICIAS]", cards_html)

    # Salvar o arquivo HTML
    diretorio_atual = os.getcwd()
    caminho_html = os.path.join(diretorio_atual, "index.html")
    
    with open(caminho_html, "w", encoding="utf-8") as arquivo:
        arquivo.write(html_final)
        
    print(f"✅ Arquivo HTML gerado com sucesso: {caminho_html}")


# -------------------------------------------------------------------
# EXECUÇÃO E ANÁLISE ESTATÍSTICA
# -------------------------------------------------------------------
if __name__ == "__main__":
    
    # Executando a extração com limites (ajuste conforme a necessidade)
    dados_noticias = extrair_noticias_rss(KEYWORDS_NOTICIAS, limite_por_termo=15)
    dados_youtube = extrair_comentarios_youtube(KEYWORDS_DEBATE, api_key=YOUTUBE_API_KEY, max_videos_por_termo=3, max_comentarios_por_video=20)
    dados_reddit = extrair_comentarios_reddit(KEYWORDS_DEBATE, client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, limite=3)
    
    todos_os_dados = dados_noticias + dados_youtube + dados_reddit
    
    if todos_os_dados:
        df = pd.DataFrame(todos_os_dados)
        
        print("\n🧠 Analisando sentimentos dos textos (LeIA)...")
        resultados = df["texto"].apply(classificar_sentimento_e_rotulo)
        df["score_sentimento"] = [r[0] for r in resultados]
        df["classificacao_preliminar"] = [r[1] for r in resultados]
        
        df["data_dt"] = pd.to_datetime(df["data"], errors='coerce', utc=True)
        df["ano"] = df["data_dt"].dt.year
        df["ano_mes"] = df["data_dt"].dt.to_period("M").astype(str)

        diretorio_atual = os.getcwd()
        
        print("\n🎨 Gerando gráficos para a análise de dados...")

        # Gráfico 1: Distribuição de Sentimento por Fonte
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=df, x="fonte_tipo", hue="classificacao_preliminar", palette="Set2")
        plt.title("Distribuição de Sentimentos por Fonte de Dados", fontsize=14, fontweight="bold")
        plt.xlabel("Fonte do Dado", fontsize=12)
        plt.ylabel("Quantidade de Registros", fontsize=12)
        plt.legend(title="Sentimento")
        plt.tight_layout()
        caminho_g1 = os.path.join(diretorio_atual, "grafico_sentimento_por_fonte.png")
        plt.savefig(caminho_g1, dpi=300)
        plt.close()

        # Gráfico 2: Evolução Temporal por Ano
        df_temporal = df.dropna(subset=["ano"]).sort_values("ano")
        if not df_temporal.empty:
            plt.figure(figsize=(12, 6))
            df_grouped = df_temporal.groupby(["ano", "classificacao_preliminar"]).size().unstack(fill_value=0)
            df_grouped.plot(kind="bar", stacked=True, colormap="tab10", figsize=(12, 6))
            plt.title("Evolução Temporal dos Discursos", fontsize=14, fontweight="bold")
            plt.xlabel("Ano de Publicação", fontsize=12)
            plt.ylabel("Volume de Registros", fontsize=12)
            plt.legend(title="Sentimento")
            plt.xticks(rotation=0)
            plt.tight_layout()
            caminho_g2 = os.path.join(diretorio_atual, "grafico_evolucao_temporal.png")
            plt.savefig(caminho_g2, dpi=300)
            plt.close()

        # Gráfico 3: Proporção Geral
        plt.figure(figsize=(8, 8))
        df["classificacao_preliminar"].value_counts().plot.pie(
            autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel")
        )
        plt.title("Proporção Geral de Sentimentos Coletados", fontsize=14, fontweight="bold")
        plt.ylabel("")
        plt.tight_layout()
        caminho_g3 = os.path.join(diretorio_atual, "grafico_distribuicao_sentimentos.png")
        plt.savefig(caminho_g3, dpi=300)
        plt.close()

        # Chama a função geradora de HTML atualizada
        gerar_dashboard_html(df)
        
        print("\n🎉 Análise descritiva, gráficos e HTML concluídos!")

    else:
        print("❌ Nenhum dado foi coletado.")