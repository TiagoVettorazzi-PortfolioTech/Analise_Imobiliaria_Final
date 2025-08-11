import streamlit as st
import pandas as pd
import os
from modules.model import load_and_train_model
import pydeck as pdk
from modules.model import data_frame


# Função para rodar o servidor FastAPI
# def run_fastapi():
#     uvicorn.run('1_🏠home.py', host="0.0.0.0", port=8000)

# # Inicia o FastAPI em uma thread separada
# thread = threading.Thread(target=run_fastapi, daemon=True)
# thread.start()

# Configuração da página
st.set_page_config(page_title="Simulador de Imóveis", layout="wide")


#sst.sidebar.title("Menu")
# Título principal
st.title("🏡 Bem-vindo ao Simulador de Imóveis")
st.write("#### Escolha uma opção abaixo para explorar os dados:")


# Carregar o modelo treinado
model, kmeans = load_and_train_model()

df = data_frame()
#st.write(df)
numericas = [
    "aream2", "Quartos", "banheiros", "vagas", "condominio", 
    "latitude", "longitude", "idh_longevidade", "area_renda", 
    "distancia_centro", "cluster_geo"
]

def exibir_scater(df):

    bins = [0, 100000, 250000, 500000, 1000000, float('inf')]
    labels = ['0-100k', '100k-250k', '250k-500k', '500k-1M', 'Acima de 1M']

    df['preco_bin'] = pd.cut(df['preco'], bins=bins, labels=labels)

    # Mapear os labels de bins para valores numéricos para usar no mapa
    bin_values = {
        '0-100k': 100000,
        '100k-250k': 250000,
        '250k-500k': 500000,
        '500k-1M': 750000,
        'Acima de 1M': 1500000
    }

    # Substituir os bins por valores numéricos
    df['preco_bin_numeric'] = df['preco_bin'].map(bin_values)

    # Preparar os dados para o mapa
    df_filtrado = df.dropna(subset=['longitude', 'latitude'])  

    # Gerar o mapa de calor
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",  
        data=df_filtrado,  
        get_position=["longitude", "latitude"], 
        get_weight="preco_bin_numeric", 
        opacity=0.8, 
        threshold=0.2  
    )

    # Definir o estado de visualização do mapa
    view_state = pdk.ViewState(
        latitude=df_filtrado["latitude"].mean(),
        longitude=df_filtrado["longitude"].mean(),
        zoom=12,
        pitch=0
    )
    st.pydeck_chart(pdk.Deck(layers=[heatmap_layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v10"))

st.write("## 📍 Mapa de calor por preço Fortaleza")
st.write('Este mapa representa a distribuição de preços dos imóveis em Fortaleza. As áreas em vermelho são as areas com imóveis mais caros. As áreas amarelas são as áreas com imóveis mais baratos.')
exibir_scater(df)

# class MultiApp:
#     def __init__(self):
#         self.apps = {}

#     def add_app(self, title, func):
#         """Adiciona uma nova página ao app"""
#         self.apps[title] = func

#     def run(self):
#         """Executa a página selecionada no menu lateral"""
#         with st.sidebar:
#             selected = option_menu(
#                 menu_title="Menu",  # Nome do menu na barra lateral
#                 options=list(self.apps.keys()),  # Opções disponíveis
#                 icons=['cloud', 'calculator'],  # Ícones para cada página
#                 menu_icon="cast",
#                 default_index=0,
#                 styles={
#                     "container": {"padding": "5px"},
#                     "nav-link": {"color": "black", "font-weight": "bold"},
#                     "nav-link-selected": {"color": "white", "background-color": "green"},
#                 }
#             )

#         # Chama a função correspondente apenas uma vez
#         if selected in self.apps:
#             self.apps[selected]()

# # Funções das páginas
# def previsao():
#     st.write("### Página de Previsão de Preços")
#     st.write("Aqui você poderá prever os preços dos imóveis.")

# def simulador():
#     st.write("### Página do Simulador de Investimentos")
#     st.write("Aqui você pode simular investimentos em imóveis.")

# # Criando a aplicação
# app = MultiApp()
# app.add_app("Previsão", previsao)
# app.add_app("Simulador", simulador)
# app.run()










def create_docx_curriculo(self, arquivo_json, arquivo_saida='curriculo.docx', logo_path='Logo2.png'):
        """Cria um documento Word formatado a partir de dados de um currículo em JSON e adiciona um logo."""
        try:
            with open(arquivo_json, 'r', encoding='utf-8') as f:
                dados = json.load(f)

            estrutura_padrao = {
                "informacoes_pessoais": {"nome": "", "cidade": "", "email": "", "telefone": "", "cargo": ""},
                "resumo_qualificacoes": [],
                "experiencia_profissional": [],
                "educacao": [],
                "certificacoes": []
            }
            dados = self.validate_json(dados, estrutura_padrao)

            doc = Document()
            estilo = doc.styles['Normal']
            estilo.font.name = 'Calibri'
            estilo.font.size = Pt(11)
            estilo.font.color.rgb = RGBColor(0, 0, 0)

            def adicionar_espaco():
                """Adiciona um parágrafo vazio para espaçamento."""
                doc.add_paragraph().paragraph_format.space_after = Pt(12)

            if logo_path:
                section = doc.sections[0]
                section.header_distance = Cm(0.6)

                header = section.header
                header_paragraph = header.paragraphs[0]
                run = header_paragraph.add_run()
                run.add_picture(logo_path, width=Inches(0.8))  # Ajusta o tamanho do logo
                header_paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.RIGHT  # Alinha à direita

            # Informações pessoais
            informacoes_pessoais = dados.get('informacoes_pessoais', {})
            nome = informacoes_pessoais.get('nome', 'Nome Não Encontrado')
            paragrafo_nome = doc.add_paragraph(nome)
            if paragrafo_nome.runs:
                nome_run = paragrafo_nome.runs[0]
                nome_run.bold = True
                nome_run.font.size = Pt(16)
            paragrafo_nome.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

            adicionar_espaco()
            contato = f"Cidade: {informacoes_pessoais.get('cidade', 'N/A')}\nEmail: {informacoes_pessoais.get('email', 'N/A')}\nTelefone: {informacoes_pessoais.get('telefone', 'N/A')}\nPosição: {informacoes_pessoais.get('cargo', 'N/A')}"
            doc.add_paragraph(contato)

            adicionar_espaco()

            # Resumo de qualificações
            doc.add_heading('Resumo de Qualificações', level=2)
            for qualificacao in dados.get('resumo_qualificacoes', []):
                doc.add_paragraph(f"- {qualificacao}")

            adicionar_espaco()

            # Experiência profissional
            doc.add_heading('Experiência Profissional', level=2)
            for experiencia in dados.get('experiencia_profissional', []):
                empresa = experiencia.get('empresa', 'Empresa Não Informada')
                cargo = experiencia.get('cargo', 'Cargo Não Informado')
                periodo = experiencia.get('periodo', 'Período Não Informado')
                local = experiencia.get('local', 'Local Não Informado')
                atividades = experiencia.get('atividades_exercidas', [])

                doc.add_paragraph(f"{empresa} ({local})", style='Heading 3')
                doc.add_paragraph(f"{cargo} - {periodo}", style='Normal')
                
                if atividades:
                    doc.add_paragraph("Atividades exercidas:", style='Normal')
                    for atividade in atividades:
                        doc.add_paragraph(f"{atividade}", style='List Bullet')

                ferramentas = experiencia.get('ferramentas', [])
                if ferramentas:
                    doc.add_paragraph("Ferramentas utilizadas:", style='Normal')
                    for ferramenta in ferramentas:
                        doc.add_paragraph(f"{ferramenta}", style='List Bullet')

            adicionar_espaco()

            # Educação
            doc.add_heading('Educação', level=2)
            for educacao in dados.get('educacao', []):
                instituicao = educacao.get('instituicao', 'Instituição Não Informada')
                curso = educacao.get('curso', 'Curso Não Informado')
                periodo = educacao.get('periodo', 'Período Não Informado')

                doc.add_paragraph(f"{instituicao}", style='Heading 3')
                doc.add_paragraph(f"{curso} - {periodo}", style='Normal')

            adicionar_espaco()

            # Certificações
            doc.add_heading('Certificações', level=2)
            for certificacao in dados.get('certificacoes', []):
                doc.add_paragraph(f"- {certificacao}", style='Normal')

            # Salvar o documento Word
            doc.save(arquivo_saida)
            print(f"Currículo salvo em {arquivo_saida}")

        except Exception as e:
            print(f"Erro ao criar documento Word: {e}")
            print(traceback.format_exc())