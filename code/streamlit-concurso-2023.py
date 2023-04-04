 
import pandas as pd
import numpy as npy
import scipy as spy
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

import datetime
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier

# https://github.com/rasbt/watermark/blob/master/watermark/watermark.py
from watermark import watermark

import yaml

font_css = """
<style>

MainMenu {
    visibility:hidden;
}

footer {
    visibility:visible;
}
footer:after {
    content:'@2023: Ibon Martínez-Arranz';
    display:block;
    position:relative;
    font-size:12px;
    color:#FE4A4A;
    padding:0px;
    top:3px;
}

h1 {
    color: #FE4A4A;
}
h2 {
    color: #FE4A4A;
}
h3 {
    color: #FE4A4A;
}
h4 {
    color: #FE4A4A;
}

.badge-custom {
    color: #CCCCCC;
    background-color: #0000CC;
}

</style>

<link rel="stylesheet" href="https://m axcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.9.1/font/bootstrap-icons.css">
"""

# FUNCTIONS

def pca_plot(df, data, project_config):
    
    df = df.merge(data[project_config['project']['variable']['name']], 
                  left_index = True, 
                  right_index = True).copy()
    df = df.reset_index()
    df.rename(columns = {project_config['project']['data']['index_colname']: 'Sample'}, inplace = True)
    
    altplot = \
        alt.Chart(df)\
            .mark_point(filled = True, size = 100)\
            .encode(
                x = alt.X('PC1', title = 'Principal Component 1'),
                y = alt.Y('PC2', title = 'Principal Component 2'),
                color = alt.Color(project_config['project']['variable']['name'],
                                  scale = alt.Scale(
                                      domain = [project_config['project']['variable']['grupo01']['name'], project_config['project']['variable']['grupo02']['name']],
                                      range = [project_config['project']['variable']['grupo01']['color'], project_config['project']['variable']['grupo02']['color']]),
                                  legend = alt.Legend(title = project_config['project']['variable']['clean_name'])),
                                  
                tooltip = [
                    alt.Tooltip('Sample'),
                    alt.Tooltip('PC1', format = '.2f'),
                    alt.Tooltip('PC2', format = '.2f')]
                )
    return altplot.properties(height = 600, width = 800).interactive()

def pcamodels(data, 
              n_components, 
              scoring = 'accuracy',
              ylabel = 'Accuracy',
              n_splits = 3,):
        
    steps = [('log', FunctionTransformer(npy.log2)),
                ('norm', StandardScaler()),
                ('pca', PCA()),
                ('model', LogisticRegression())]
    

    # Grid de hiperparámetros evaluados
    # =================================
    param_grid = {'pca__n_components': range(1, n_components + 1)}

    grid = GridSearchCV(
            estimator  = Pipeline(steps = steps),
            param_grid = param_grid,
            scoring    = scoring,
            n_jobs     = -1,
            cv         = StratifiedKFold(n_splits = n_splits, 
                                         shuffle = False, 
                                         random_state = 123), 
            refit      = True,
            verbose    = 0,
            return_train_score = True
        )
    
    #st.write(data[project_config['project']['variable']['name']])

    grid.fit(X = data[profile.columns], 
             y = data[project_config['project']['variable']['name']])
    
    resultados = pd.DataFrame(grid.cv_results_)

    # Gráfico resultados validación cruzada para cada hiperparámetro
    # ==============================================================
    fig, ax = plt.subplots(nrows = 1, ncols = 1, 
                            figsize = (7, 3.84), sharey = True)

    resultados.plot('param_pca__n_components', 'mean_train_score', ax = ax)
    resultados.plot('param_pca__n_components', 'mean_test_score', ax = ax)
    ax.fill_between(resultados.param_pca__n_components.astype(npy.float),
                    resultados['mean_train_score'] + resultados['std_train_score'],
                    resultados['mean_train_score'] - resultados['std_train_score'],
                    alpha = 0.2)
    ax.fill_between(resultados.param_pca__n_components.astype(npy.float),
                    resultados['mean_test_score'] + resultados['std_test_score'],
                    resultados['mean_test_score'] - resultados['std_test_score'],
                    alpha = 0.2)
   
    L = ax.legend(loc = 'upper left', frameon = False)
    L.get_texts()[0].set_text('Average Train Score')
    L.get_texts()[1].set_text('Average Test Score')
    
    ax.set_title('')
    ax.set_xlabel('Número de Componentes')
    ax.set_ylabel(ylabel)
    
    return fig, resultados, grid



@st.experimental_memo
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


# SIDEBAR CONFIGURATION

f = open("about.txt", "r")

st.set_page_config(
     page_title = "Análisis de Componentes Principales y Modelización",
     page_icon = "https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f916.svg", # https://twemoji-cheatsheet.vercel.app/
     layout = "wide",
     initial_sidebar_state = "expanded",
     menu_items = {
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': f.read()
     }
)                                    
f.close()

st.write(font_css, unsafe_allow_html = True)

#st.sidebar.title("Análisis de Componentes Principales y Modelización")
st.sidebar.image("../images/logo-streamlit-concurso-2023.png")

st.sidebar.info("""
El Análisis de Componentes Principales (PCA) es una poderosa técnica estadística que se usa ampliamente en el aprendizaje automático para una variedad de tareas, como la visualización de datos, la selección de características y el preprocesamiento de datos.""")

st.sidebar.markdown('<h4 class="badge badge-pill badge-primary"> <i class="bi bi-github"> imarranz </i></h4>', unsafe_allow_html=True)


st.title(":bar_chart: Análisis de Componentes Principales y Modelización")

st.markdown("""
EL Análisis de Componentes Principales (PCA) es una valiosa técnica de preprocesamiento en el modelado predictivo. Puede ayudar en el análisis de datos exploratorios y la detección de valores atípicos, y también reduce la dimensionalidad cuando el número de variables es mayor que el tamaño de la muestra (d>n). Además, PCA se usa comúnmente en conjuntos de datos con variables altamente redundantes o correlacionadas, lo que puede generar multicolinealidad e inestabilidad en los modelos de regresión. La multicolinealidad infla la varianza de las estimaciones de los parámetros, lo que puede hacer que sean estadísticamente insignificantes cuando deberían ser significativos (Kerns 2010). En las siguientes secciones, utilizaremos PCA como un paso de preprocesamiento y lo combinaremos con un modelo de regresión logística regularizado en L2 para abordar un problema de clasificación.

## Modelo PCA

Los modelos PCA se usan comúnmente para dos propósitos principales: 1) para reducir las dimensiones de un conjunto de datos para facilitar el análisis y 2) para identificar de manera eficiente las fuentes de error. Describir un proceso complejo en términos de unas pocas variables puede ser más manejable que considerar todas las variables que interactúan dentro de él. Para lograr esto, un modelo PCA crea nuevas variables, llamadas Componentes Principales (PC), que explican la mayor parte de la actividad en el proceso. Las PC están etiquetadas como t0, t1, t2, etc., y t0 explica la mayor parte de la variación en el proceso. La variable que tiene el mayor efecto sobre el componente principal es la que requiere mayor investigación para reducir la variación del proceso. Las PC no están correlacionadas con otras variables y no tienen unidades de medida.

Los modelos PCA usan solo las variables de entrada del proceso y no se seleccionan objetivos de modelo al crear el modelo. Las PC sirven como una herramienta eficaz para el análisis exploratorio de datos y la detección de valores atípicos, así como para la reducción de la dimensionalidad cuando el número de variables es mayor que el tamaño de la muestra (d>n). Reducir las dimensiones de un conjunto de datos es particularmente útil para conjuntos de datos con variables altamente redundantes o correlacionadas, lo que puede causar inestabilidad en los modelos de regresión debido a la multicolinealidad. En tales casos, la información redundante puede inflar la varianza de las estimaciones de los parámetros y hacerlas estadísticamente insignificantes cuando, de otro modo, habrían sido significativas. En secciones posteriores, aplicaremos PCA como técnica de preprocesamiento de datos y la combinaremos con un modelo de regresión logística regularizado en L2 para resolver un problema de clasificación.""")


with st.form("configuración"):
    
    uf = st.file_uploader(label = 'Selecciona el Fichero de Configuración:',
                          type = 'yaml',
                          accept_multiple_files = False)
    
    if uf is not None:
        
        with open(uf.name, "r") as yamlfile:
            project_config = yaml.load(yamlfile, Loader = yaml.FullLoader)
        
        data = pd.read_excel(project_config['project']['path'], 
                             sheet_name = project_config['project']['data']['sheet_name'], 
                             index_col = project_config['project']['data']['index_colname'])
      
        grupo = st.selectbox(label = 'Variable', 
                             options = data.columns, 
                             index = list(data.columns).index(project_config['project']['variable']['name']))
        
        opciones = list(data[grupo].unique())
        
        col1, col2 = st.columns(2, gap = 'small')
        
        with col1:
            
            g1 = st.selectbox(label = 'Grupo A', 
                              options = opciones,
                              index = opciones.index(project_config['project']['variable']['grupo01']['name']))
            g1_color = st.color_picker('Color:', 
                                       project_config['project']['variable']['grupo01']['color']) 
            project_config['project']['variable']['grupo01']['color'] = g1_color
            
        with col2:
            
            g2 = st.selectbox(label = 'Grupo B', 
                              options = opciones,
                              index = opciones.index(project_config['project']['variable']['grupo02']['name'])) 
            g2_color = st.color_picker('Color:', 
                                       project_config['project']['variable']['grupo02']['color'])
            project_config['project']['variable']['grupo02']['color'] = g2_color
            
        with st.expander("Ver Configuración del Proyecto"):
            st.json(project_config)
        
        with open("CONFIG_.yaml", "r") as yamlfile:
            CONFIG_ = yaml.load(yamlfile, Loader = yaml.FullLoader)

    submitted = st.form_submit_button("Inicio del Análisis", )
    
if submitted:
    
    data_show_container = st.container()
    
    profile = data.drop(columns = project_config['project']['variable']['name']).dropna().copy()
   
    with data_show_container:
        
        st.markdown("## Datos")
        
        tabs1, tabs2 = st.tabs(['Datos', 'Descripción'])
        
        with tabs1:
            
            st.table(data.head())
            
            st.download_button(
                label = "Descargar (csv)",
                data = convert_df(data),
                file_name = time.strftime("%Y%m%d%H%M%S", time.gmtime()) + "_data.csv",
                mime = "text/csv",
                key = 'download-data-csv')
            
        with tabs2:
            
            st.table(profile.describe().T) 
            
    pca_show_container = st.container()
    
    with pca_show_container:

        st.markdown("## Análisis de Componentes Principales")
        
        N_COMPONENTS = npy.min(list(profile.shape) + [project_config['model']['n_max_components']])

        steps = [('log', FunctionTransformer(npy.log2)),
                 ('norm', StandardScaler()),
                 ('pca', PCA(n_components = N_COMPONENTS)),]
        
        pca_data = Pipeline(steps = steps).fit(profile)
        
        tabs1, tabs2, tabs3, tabs4, tabs5, tab6 = \
            st.tabs(['Varianza Explicada', 'Componentes',
                     'Gráfico PCA', 
                     'Cargas',
                     'Gráfico de Cargas',
                     'Importancia de las Componentes'])
        
        pca_df = pd.DataFrame(
            pca_data.transform(profile),
            columns = ['PC'+str(i) for i in range(1, N_COMPONENTS + 1)],
            index = profile.index)            
        
        with tabs1:
            
            modelo_pca = pca_data.named_steps['pca']
                
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 4.5))
            ax.bar(
                x = npy.arange(modelo_pca.n_components_) + 1,
                height = modelo_pca.explained_variance_ratio_
            )

            for x, y in zip(npy.arange(len(pca_df.columns)) + 1, modelo_pca.explained_variance_ratio_):
                
                label = round(y, 2)
                ax.annotate(
                    label,
                    (x,y),
                    textcoords = "offset points",
                    xytext = (0,10),
                    color = 'C0',
                    fontsize = 6,
                    ha = 'center'
                )

            prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
                
            ax.plot(
                npy.arange(modelo_pca.n_components_) + 1,
                prop_varianza_acum,
                marker = 'o',
                color = 'red',
            )

            for x, y in zip(npy.arange(modelo_pca.n_components_) + 1, prop_varianza_acum):
                label = round(y, 2)
                ax.annotate(
                    label,
                    (x,y),
                    textcoords = "offset points",
                    color = 'red',
                    xytext = (0,10),
                    fontsize = 6,
                    ha = 'center'
                )
            ax.grid(axis = 'y', linestyle = ':', color ='0.6')
            ax.set_ylim(0,1.1)
            ax.set_xticks(npy.arange(modelo_pca.n_components_) + 1)
            ax.set_title('Porcentaje de Varianza Explicada por cada Componente')
            ax.set_xlabel('Componentes Principales')
            ax.set_ylabel('Varianza Explicada (%)');
            
            st.pyplot(fig)
            
            pca_variance = \
                pd.DataFrame({
                    'PC': pca_df.columns,
                    'Varianza': modelo_pca.explained_variance_ratio_,
                    'Varianza Acumulada': modelo_pca.explained_variance_ratio_.cumsum()})
            
            st.write(pca_variance.style.hide_index())
            
            st.download_button(
                label = "Descargar (csv)",
                data = convert_df(pca_variance),
                file_name = time.strftime("%Y%m%d%H%M%S", time.gmtime()) + "_pca_varianza.csv",
                mime = "text/csv",
                key = 'download-pca-variance-csv') 

        with tabs2:
            
            pca_df = pd.DataFrame(
                pca_data.transform(profile),
                columns = ['PC'+str(i) for i in range(1, N_COMPONENTS+1)],
                index = profile.index)
            
            st.write(pca_df.style.format('{:.2f}').background_gradient(cmap = project_config['project']['cmap']))
            
            st.download_button(
                label = "Descargar (csv)",
                data = convert_df(pca_df),
                file_name = time.strftime("%Y%m%d%H%M%S", time.gmtime()) + "_pca_componentes.csv",
                mime = "text/csv",
                key = 'download-pca-components-csv') 
            
        with tabs3:
            
            st.altair_chart(pca_plot(pca_df, data, project_config))
            
        with tabs4:
            
            loadings = pd.DataFrame(pca_data.named_steps['pca'].components_.T)
            loadings.index = profile.columns
            loadings.columns = pca_df.columns
            
            st.write(loadings\
                        .style\
                            .format('{:.2f}')\
                            .background_gradient(cmap = project_config['project']['cmap']))
            
            st.download_button(
                label = "Descargar (csv)",
                data = convert_df(loadings),
                file_name = time.strftime("%Y%m%d%H%M%S", time.gmtime()) + "_pca_loadings.csv",
                mime = "text/csv",
                key = 'download-pca-loadings-csv') 
            
        with tabs5:
            
            loadings = \
            loadings[['PC1', 'PC2']]\
                .reset_index()\
                .rename(columns = {'index':'ID'})
            
            altloadingsplot = \
                alt.Chart(loadings)\
                    .mark_point(filled = False)\
                        .encode(
                            x = alt.X('PC1', title = 'Principal Component 1'),
                            y = alt.Y('PC2', title = 'Principal Component 2'),
                            tooltip = [
                                alt.Tooltip('ID'),])\
                        .properties(height = 600, width = 800)\
                        .interactive()
                    
            st.altair_chart(altloadingsplot)
        
        with tab6:
            
            data_importance_features = \
                pca_df.merge(data[project_config['project']['variable']['name']],
                             left_index = True, 
                             right_index = True)

            forest = RandomForestClassifier(random_state = 0)
            X = data_importance_features.drop(columns = project_config['project']['variable']['name'])
            y = data_importance_features[project_config['project']['variable']['name']]
            forest.fit(X,y)
            importances = forest.feature_importances_
            std = npy.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)
            forest_importances = pd.Series(importances, index = X.columns)

            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))
            forest_importances.plot.bar(yerr = std, ax = ax)
            ax.set_title("Características importantes usando MDI")
            ax.set_ylabel("Disminución Media de la Impureza")
            fig.tight_layout()
            
            
            col1, col2 = st.columns([3,1], gap = 'small')
            
            with col1:
                st.pyplot(fig)
            with col2:
                st.write(
                    pd.DataFrame(
                        {'Promedio': importances,
                         'dt': std},
                        index = X.columns))
                       

    modelling_show_container = st.container()

    with modelling_show_container:

        st.markdown("## Modelización")
        
        st.markdown("""
            El Análisis de Componentes Principales (PCA) es una poderosa técnica estadística que se usa ampliamente en el aprendizaje automático para una variedad de tareas, como la visualización de datos, la selección de características y el preprocesamiento de datos. PCA es particularmente útil cuando se trabaja con grandes conjuntos de datos que tienen una gran cantidad de características o variables, ya que puede acelerar significativamente el tiempo de procesamiento de los algoritmos de aprendizaje automático.

            PCA funciona al identificar la estructura subyacente en los datos y reducir la cantidad de dimensiones o variables, al tiempo que conserva la información esencial necesaria para representar los datos. Esto se logra transformando los datos originales en un nuevo conjunto de variables no correlacionadas llamadas componentes principales. Cada componente principal representa una combinación lineal de las variables originales, y los primeros componentes principales suelen capturar la mayor parte de la variación en los datos.

            Al reducir la cantidad de dimensiones, PCA puede ayudar a superar la maldición de la dimensionalidad, que es un problema común en el aprendizaje automático donde la cantidad de variables en un conjunto de datos crece más que la cantidad de observaciones, lo que lleva a un sobreajuste y un rendimiento predictivo reducido. PCA también puede ayudar a eliminar el ruido y la redundancia en los datos, lo que da como resultado predicciones más precisas y confiables.

            Además de mejorar el rendimiento del aprendizaje automático, PCA también puede proporcionar información sobre la estructura subyacente de los datos, revelando patrones y relaciones que pueden no ser evidentes solo con las variables originales. Esto hace que PCA sea una herramienta valiosa para el análisis exploratorio de datos y la visualización de datos.

            En general, PCA es una técnica versátil y poderosa que se puede aplicar a una amplia gama de problemas de aprendizaje automático, desde regresión y clasificación hasta agrupación y detección de anomalías. Al reducir la dimensionalidad de los datos, PCA puede acelerar el tiempo de procesamiento, mejorar la precisión y la confiabilidad, y brindar información sobre la estructura subyacente de los datos.
            """)
        
        data_to_modelate = data.dropna().copy()
            
        data_to_modelate.replace(
            {project_config['project']['variable']['grupo01']['name']: 0,
             project_config['project']['variable']['grupo02']['name']: 1,},
            inplace = True)        
        
        # https://scikit-learn.org/stable/modules/model_evaluation.html
        
        tabs1, tabs2, tabs3, tabs4, tabs5, tabs6, tabs7 = \
            st.tabs(['Accuracy', 'Average Precision', 'F1', 
                     'Cross-Entropy Loss',
                     'Precision', 'Recall', 'ROC AUC'])
        
        with tabs1:
        
            st.success(CONFIG_['metricas']['exactitud'])
        
            fig, resultados, grid = \
            pcamodels(data = data_to_modelate, 
                    n_components = N_COMPONENTS, 
                    scoring = 'accuracy',
                    ylabel = 'Accuracy',
                    n_splits = project_config['model']['n_splits'],)
        
            col1, col2 = st.columns(2, gap = 'small')        
        
            with col1:
                st.pyplot(fig)
                st.info(f"""Optimum number of components: {grid.best_params_['pca__n_components']}  
                        Best Score: {grid.best_score_:.2f}""")

            with col2:
                st.write(
                    resultados.filter(regex = '(param.*|mean_t|std_t)') \
                        .drop(columns = 'params') \
                        .sort_values('mean_test_score', ascending = False)) 

        with tabs2:
        
            st.success(CONFIG_['metricas']['precision'])
        
            fig, resultados, grid = \
            pcamodels(data = data_to_modelate, 
                    n_components = N_COMPONENTS, 
                    scoring = 'average_precision',
                    ylabel = 'Average Precision',
                    n_splits = project_config['model']['n_splits'],)
        
            col1, col2 = st.columns(2, gap = 'small')        
        
            with col1:
                st.pyplot(fig)
                st.info(f"""Optimum number of components: {grid.best_params_['pca__n_components']}  
                        Best Score: {grid.best_score_:.2f}""")

            with col2:
                st.write(
                    resultados.filter(regex = '(param.*|mean_t|std_t)') \
                        .drop(columns = 'params') \
                        .sort_values('mean_test_score', ascending = False)) 

        with tabs3:
        
            st.success(CONFIG_['metricas']['f1'])
        
            fig, resultados, grid = \
            pcamodels(data = data_to_modelate, 
                    n_components = N_COMPONENTS, 
                    scoring = 'f1',
                    ylabel = 'F1',
                    n_splits = project_config['model']['n_splits'],)
        
            col1, col2 = st.columns(2, gap = 'small')        
        
            with col1:
                st.pyplot(fig)
                st.info(f"""Optimum number of components: {grid.best_params_['pca__n_components']}  
                        Best Score: {grid.best_score_:.2f}""")

            with col2:
                st.write(
                    resultados.filter(regex = '(param.*|mean_t|std_t)') \
                        .drop(columns = 'params') \
                        .sort_values('mean_test_score', ascending = False)) 

        with tabs4:
        
            st.success(CONFIG_['metricas']['entropia'])
        
            fig, resultados, grid = \
            pcamodels(data = data_to_modelate, 
                    n_components = N_COMPONENTS, 
                    scoring = 'neg_log_loss',
                    ylabel = 'Cross-Entropy Loss',
                    n_splits = project_config['model']['n_splits'],)
        
            col1, col2 = st.columns(2, gap = 'small')        
        
            with col1:
                st.pyplot(fig)
                st.info(f"""Optimum number of components: {grid.best_params_['pca__n_components']}  
                        Best Score: {grid.best_score_:.2f}""")

            with col2:
                st.write(
                    resultados.filter(regex = '(param.*|mean_t|std_t)') \
                        .drop(columns = 'params') \
                        .sort_values('mean_test_score', ascending = False)) 

        with tabs5:
            
            st.success(CONFIG_['metricas']['precision'])
            
            fig, resultados, grid = \
            pcamodels(data = data_to_modelate, 
                    n_components = N_COMPONENTS, 
                    scoring = 'precision',
                    ylabel = 'Precision',
                    n_splits = project_config['model']['n_splits'],)

            col1, col2 = st.columns(2, gap = 'small')        
        
            with col1:
                st.pyplot(fig)
                st.info(f"""Optimum number of components: {grid.best_params_['pca__n_components']}  
                        Best Score: {grid.best_score_:.2f}""")

            with col2:
                st.write(
                    resultados.filter(regex = '(param.*|mean_t|std_t)') \
                        .drop(columns = 'params') \
                        .sort_values('mean_test_score', ascending = False))  

        with tabs6:
            
            st.success(CONFIG_['metricas']['recall'])
            
            fig, resultados, grid = \
            pcamodels(data = data_to_modelate, 
                    n_components = N_COMPONENTS, 
                    scoring = 'recall',
                    ylabel = 'Recall',
                    n_splits = project_config['model']['n_splits'],)

            col1, col2 = st.columns(2, gap = 'small')        
        
            with col1:
                st.pyplot(fig)
                st.info(f"""Optimum number of components: {grid.best_params_['pca__n_components']}  
                        Best Score: {grid.best_score_:.2f}""")

            with col2:
                st.write(
                    resultados.filter(regex = '(param.*|mean_t|std_t)') \
                        .drop(columns = 'params') \
                        .sort_values('mean_test_score', ascending = False))                  

        with tabs7:
            
            st.success("""Calcular el área bajo la curva característica operativa del receptor (ROC AUC) a partir de las puntuaciones de predicción.""")
            
            fig, resultados, grid = \
            pcamodels(data = data_to_modelate, 
                    n_components = N_COMPONENTS, 
                    scoring = 'roc_auc',
                    ylabel = 'Area Under the Curve',
                    n_splits = project_config['model']['n_splits'],)

            col1, col2 = st.columns(2, gap = 'small')        
        
            with col1:
                st.pyplot(fig)
                st.info(f"""Número Óptimo de Componentes: {grid.best_params_['pca__n_components']}  
                        Mejor Puntuación: {grid.best_score_:.2f}""")
                st.write(grid)                
                
            with col2:
                st.write(
                    resultados.filter(regex = '(param.*|mean_t|std_t)') \
                        .drop(columns = 'params') \
                        .sort_values('mean_test_score', ascending = False))   

    sesion_show_container = st.container()

    with sesion_show_container:

        st.markdown("## Configuración de la Sesión")
        
        st.text(
            watermark(
                author = project_config['project']['author'],
                updated = True,
                current_date = True,
                current_time = True,
                python = True,
                machine = True,
                packages = "pandas,numpy,scipy,streamlit,matplotlib,seaborn,altair,sklearn,yaml"))
        
    bibliography_show_container = st.container()

    with bibliography_show_container:

        st.markdown("## Bibliografía")

        st.markdown("""
  * Introduction to Machine Learning with Python: A Guide for Data Scientists

  * Python Data Science Handbook by Jake VanderPlas

  * Linear Models with R by Julian J.Faraway

  * An Introduction to Statistical Learning: with Applications in R (Springer Texts in Statistics)

  * OpenIntro Statistics: Fourth Edition by David Diez, Mine Çetinkaya-Rundel, Christopher Barr

  * Points of Significance Principal component analysis by Jake Lever, Martin Krzywinski & Naomi Altman""")
