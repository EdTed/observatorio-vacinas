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
from google import genai

# Configuração visual dos gráficos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

analyzer = SentimentIntensityAnalyzer()

# CHAVES PUXADAS DO GITHUB SECRETS (ou hardcoded para testes locais)
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 
KEYWORDS_NOTICIAS = ["vacina", "vacinação", "antivax", "movimento antivacina", "imunização"]
KEYWORDS_DEBATE = ["vacina covid", "antivax brasil", "vacinação obrigatória", "efeito colateral vacina"]

# -------------------------------------------------------------------
# FUNÇÃO DE RESUMO GERAL COM IA (CAIXAS DO TOPO)
# -------------------------------------------------------------------
def gerar_resumo_ia(df_textos, sentimento):
    if not GEMINI_API_KEY:
        return "⚠️ Chave do Gemini não configurada."
    
    if df_textos.empty:
        return f"Não há comentários suficientes para gerar um resumo {sentimento}."

    # Tenta até 3 vezes caso o servidor do Google esteja ocupado (Erro 503 ou 429)
    for tentativa in range(3):
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            textos_combinados = " \n- ".join(df_textos['texto'].astype(str).head(50).tolist())
            
            prompt = f"""
            Você é um analista de dados. Analise os seguintes comentários/notícias classificados como {sentimento}s sobre vacinas.
            Crie um resumo direto de, NO MÁXIMO, 3 linhas explicando os principais pontos levantados. Sem saudações.
            
            Textos:
            - {textos_combinados}
            """
            
            # Usando o modelo 'gemini-3.6-flash', que é o mais estável disponível atualmente
        

            response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        
            )
            
            return response.text.strip()
        
        except Exception as e:
            if tentativa < 2:
                time.sleep(12)  # Aguarda 5 segundos antes de tentar novamente
                continue
            else:
                # Se após 3 tentativas falhar, retorna uma mensagem amigável em vez do código de erro
                return "Resumo temporariamente indisponível devido a alta demanda no servidor."
                

# -------------------------------------------------------------------
# MÓDULOS DE EXTRAÇÃO DE DADOS
# -------------------------------------------------------------------
def extrair_noticias_rss(termos, limite_por_termo=500):
    noticias = []
    print("🌐 [1/3] Coletando notícias...")
    for termo in termos:
        termo_encoded = quote(termo)
        rss_url = f"https://news.google.com/rss/search?q={termo_encoded}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:limite_por_termo]:
            url = entry.link
            texto = entry.title
            noticias.append({
                "fonte_tipo": "Notícia Portal", "termo_busca": termo,
                "titulo": entry.title, "texto": texto,
                "data": entry.published, "url": url
            })
    return noticias

def extrair_comentarios_youtube(termos, api_key, max_videos=100, max_coments=25):
    comentarios = []
    if not api_key: return comentarios
    print("🎥 [2/3] Coletando YouTube...")
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        for termo in termos:
            search_response = youtube.search().list(q=termo, part="id,snippet", maxResults=max_videos, type="video", relevanceLanguage="pt").execute()
            for item in search_response.get("items", []):
                try:
                    c_resp = youtube.commentThreads().list(part="snippet", videoId=item["id"]["videoId"], maxResults=max_coments, textFormat="plainText").execute()
                    for c_item in c_resp.get("items", []):
                        txt = c_item["snippet"]["topLevelComment"]["snippet"]["textDisplay"].strip()
                        if len(txt) > 10:
                            comentarios.append({"fonte_tipo": "Comentário YouTube", "termo_busca": termo, "titulo": f"Vídeo: {item['snippet']['title']}", "texto": txt, "data": c_item["snippet"]["topLevelComment"]["snippet"]["publishedAt"], "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"})
                except: continue
    except: pass
    return comentarios

def extrair_comentarios_reddit(termos, client_id, client_secret, limite=3):
    comentarios = []
    if not client_id or not client_secret: return comentarios
    print("💬 [3/3] Coletando Reddit...")
    try:
        reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent="VacinaScraper/1.0")
        for termo in termos:
            for sub in reddit.subreddit("brasil+brasilivre+conversas").search(termo, limit=limite):
                comentarios.append({"fonte_tipo": "Post Fórum", "termo_busca": termo, "titulo": sub.title, "texto": sub.selftext or sub.title, "data": pd.to_datetime(sub.created_utc, unit='s').strftime('%Y-%m-%d %H:%M:%S'), "url": f"https://reddit.com{sub.permalink}"})
                sub.comments.replace_more(limit=0)
                for c in sub.comments[:10]:
                    if len(c.body.strip()) > 15:
                        comentarios.append({"fonte_tipo": "Comentário Fórum", "termo_busca": termo, "titulo": f"Re: {sub.title[:30]}...", "texto": c.body, "data": pd.to_datetime(c.created_utc, unit='s').strftime('%Y-%m-%d %H:%M:%S'), "url": f"https://reddit.com{c.permalink}"})
    except: pass
    return comentarios

def classificar_sentimento_e_rotulo(texto):
    scores = analyzer.polarity_scores(texto)
    c = scores["compound"]
    tl = texto.lower()
    termos = ["vachina", "experimento", "chips", "forçar vacina", "ditadura sanitária", "miocardite", "trombo"]
    eh_antivax = any(t in tl for t in termos)
    if c >= 0.05: return c, "Favorável / Positivo"
    elif c <= -0.05: return c, "Discurso Crítico / Antivax Prob" if eh_antivax else "Crítico / Negativo"
    else: return c, "Neutro / Informativo"

# -------------------------------------------------------------------
# GERAÇÃO DA PÁGINA HTML (COM CARDS DE TAMANHOS UNIFORMES)
# -------------------------------------------------------------------
def gerar_dashboard_html(df, resumo_pos, resumo_neg):
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
            
            /* CAIXAS DE RESUMO IA */
            .ai-summary-grid { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 3rem; }
            .ai-box { flex: 1; min-width: 300px; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: white; }
            .ai-box h3 { margin-top: 0; font-size: 1.3rem; display: flex; align-items: center; gap: 10px; }
            .ai-pos { background: linear-gradient(135deg, #2b9348, #55a630); }
            .ai-neg { background: linear-gradient(135deg, #d90429, #ef233c); }
            .ai-box p { font-size: 1.05rem; font-weight: 500; line-height: 1.5; margin-bottom: 0;}

            /* Gráficos e Listas */
            .charts-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 4rem; }
            .chart-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; }
            .chart-card img { max-width: 100%; height: auto; border-radius: 4px; }
            
            .top-lists-wrapper { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 4rem; align-items: flex-start; }
            .top-list-box { flex: 1; min-width: 350px; background: transparent; padding: 0; }
            
            .title-pos { color: #2b9348; border-bottom: 2px solid #2b9348; padding-bottom: 15px; background: white; padding: 20px; border-radius: 8px 8px 0 0; margin: 0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
            .title-neg { color: #d90429; border-bottom: 2px solid #d90429; padding-bottom: 15px; background: white; padding: 20px; border-radius: 8px 8px 0 0; margin: 0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
            
            .top-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 15px; margin-top: 15px; }
            .top-list li { display: flex; }

            /* CARDS COM ALTURA E TAMANHO UNIFORMES */
            .news-card { 
                background: white; 
                padding: 20px; 
                border-radius: 8px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.04); 
                border: 1px solid #eaeaea;
                width: 100%;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                height: 220px; 
                box-sizing: border-box;
            }
            
            .news-card h4 { 
                margin: 8px 0; 
                font-size: 1rem; 
                color: #222;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
                overflow: hidden;
            }
            
            .news-card p { 
                font-size: 0.9rem; 
                color: #555; 
                margin: 5px 0; 
                line-height: 1.4;
                display: -webkit-box;
                -webkit-line-clamp: 3;
                -webkit-box-orient: vertical;
                overflow: hidden;
            }
            
            /* Utilitários */
            .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; color: white; text-transform: uppercase; }
            .badge-positivo { background-color: #2b9348; }
            .badge-negativo { background-color: #d90429; }
            .badge-fonte { background-color: #333; margin-right: 5px;}
            .read-more { color: #0056b3; text-decoration: none; font-weight: bold; font-size: 0.9rem; display: inline-block; margin-top: 5px;}
            footer { background-color: #222; color: white; text-align: center; padding: 2rem 0; margin-top: 3rem; }
        </style>
    </head>
    <body>
        <header>
            <div class="logo"><h1>🔬 Observatório de Saúde: Vacinas</h1></div>
            <div style="color: #666; font-weight: bold;">Monitoramento e NLP</div>
        </header>

        <div class="container">
            
            <!-- CAIXAS DA INTELIGÊNCIA ARTIFICIAL -->
            <h2 class="section-title">✨ Resumo Gerado por IA (Últimos 7 dias)</h2>
            <div class="ai-summary-grid">
                <div class="ai-box ai-pos">
                    <h3>👍 Principais Opiniões Positivas</h3>
                    <p>[TEXTO_RESUMO_POS]</p>
                </div>
                <div class="ai-box ai-neg">
                    <h3>⚠️ Principais Críticas e Preocupações</h3>
                    <p>[TEXTO_RESUMO_NEG]</p>
                </div>
            </div>

            <!-- Gráficos Gerados -->
            <h2 class="section-title">Análise de Dados e Sentimentos</h2>
            <div class="charts-grid">
                <div class="chart-card"><img src="grafico_sentimento_por_fonte.png"></div>
                <div class="chart-card"><img src="grafico_evolucao_temporal.png"></div>
                <div class="chart-card"><img src="grafico_distribuicao_sentimentos.png"></div>
            </div>

            <!-- Caixas de TOP 10 COM CARDS INDIVIDUAIS UNIFORMES -->
            <h2 class="section-title">Destaques da Semana</h2>
            <div class="top-lists-wrapper">
                <div class="top-list-box">
                    <h3 class="title-pos">Top 10 Mais Positivos</h3>
                    <ul class="top-list">[LISTA_POSITIVAS]</ul>
                </div>
                <div class="top-list-box">
                    <h3 class="title-neg">Top 10 Mais Negativos</h3>
                    <ul class="top-list">[LISTA_NEGATIVAS]</ul>
                </div>
            </div>
        </div>

        <footer><p>&copy; 2026 Observatório de Saúde.</p></footer>
    </body>
    </html>
    """

    hoje = pd.Timestamp.utcnow()
    df_7_dias = df[df["data_dt"] >= (hoje - pd.Timedelta(days=7))]

# --- NOVO: REMOVE DUPLICATAS PELO LINK OU PELO TEXTO ---
    
    df_7_dias = df_7_dias.drop_duplicates(subset=["texto"])

    top_pos = df_7_dias.sort_values(by="score_sentimento", ascending=False).head(10)
    top_neg = df_7_dias.sort_values(by="score_sentimento", ascending=True).head(10)

    def formatar_lista(df_sub, badge_cls):
        if df_sub.empty: return "<li><div class='news-card'><p><i>Nenhuma publicação encontrada.</i></p></div></li>"
        html = ""
        for _, row in df_sub.iterrows():
            titulo = str(row['titulo'])[:90] + "..." if len(str(row['titulo']))>90 else str(row['titulo'])
            texto = str(row['texto'])

            html += f"""
            <li>
                <div class="news-card">
                    <div>
                        <span class="badge {badge_cls}">Score: {row['score_sentimento']:.2f}</span>
                        <span class="badge badge-fonte">{row['fonte_tipo']}</span>
                        <h4>{titulo}</h4>
                        <p>{texto}</p>
                    </div>
                    <a href="{row['url']}" target="_blank" class="read-more">Ler na íntegra &rarr;</a>
                </div>
            </li>
            """
        return html

    # Injetar os dados na página HTML
    html_final = template_html.replace("[TEXTO_RESUMO_POS]", resumo_pos)
    html_final = html_final.replace("[TEXTO_RESUMO_NEG]", resumo_neg)
    html_final = html_final.replace("[LISTA_POSITIVAS]", formatar_lista(top_pos, "badge-positivo"))
    html_final = html_final.replace("[LISTA_NEGATIVAS]", formatar_lista(top_neg, "badge-negativo"))

    caminho_html = os.path.join(os.getcwd(), "index.html")
    with open(caminho_html, "w", encoding="utf-8") as f:
        f.write(html_final)
    print("✅ Arquivo HTML gerado com sucesso!")


if __name__ == "__main__":
    d_noticias = extrair_noticias_rss(KEYWORDS_NOTICIAS, limite_por_termo=15)
    d_youtube = extrair_comentarios_youtube(KEYWORDS_DEBATE, api_key=YOUTUBE_API_KEY, max_videos=3, max_coments=20)
    d_reddit = extrair_comentarios_reddit(KEYWORDS_DEBATE, client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, limite=3)
    
    dados = d_noticias + d_youtube + d_reddit
    
    if dados:
        df = pd.DataFrame(dados)
        resultados = df["texto"].apply(classificar_sentimento_e_rotulo)
        df["score_sentimento"] = [r[0] for r in resultados]
        df["classificacao_preliminar"] = [r[1] for r in resultados]
        df["data_dt"] = pd.to_datetime(df["data"], errors='coerce', utc=True)
        df["ano"] = df["data_dt"].dt.year
        
        print("\n🧠 Acionando Gemini API para resumos...")
        hoje = pd.Timestamp.utcnow()
        df_7_dias = df[df["data_dt"] >= (hoje - pd.Timedelta(days=7))]
        
        df_pos = df_7_dias[df_7_dias['classificacao_preliminar'].str.contains("Positivo")]
        df_neg = df_7_dias[df_7_dias['classificacao_preliminar'].str.contains("Crítico|Antivax")]
        
        resumo_positivo = gerar_resumo_ia(df_pos.sort_values(by="score_sentimento", ascending=False), "positivo")
        resumo_negativo = gerar_resumo_ia(df_neg.sort_values(by="score_sentimento", ascending=True), "negativo")
        
        # Geração de gráficos
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x="fonte_tipo", hue="classificacao_preliminar", palette="Set2")
        plt.tight_layout()
        plt.savefig("grafico_sentimento_por_fonte.png", dpi=300)
        plt.close()

        df_temporal = df.dropna(subset=["ano"]).sort_values("ano")
        if not df_temporal.empty:
            plt.figure(figsize=(12, 6))
            df_grouped = df_temporal.groupby(["ano", "classificacao_preliminar"]).size().unstack(fill_value=0)
            df_grouped.plot(kind="bar", stacked=True, colormap="tab10", figsize=(12, 6))
            plt.tight_layout()
            plt.savefig("grafico_evolucao_temporal.png", dpi=300)
            plt.close()

        plt.figure(figsize=(8, 8))
        df["classificacao_preliminar"].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette("pastel"))
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig("grafico_distribuicao_sentimentos.png", dpi=300)
        plt.close()

        gerar_dashboard_html(df, resumo_positivo, resumo_negativo)
