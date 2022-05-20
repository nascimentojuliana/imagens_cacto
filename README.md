cactus
==============================

Consta de um repositório que busca identificar se há cacto em uma imagem ou não. 

Para avaliar o impacto das mudanças climáticas na flora e na fauna da Terra, é vital quantificar como as atividades humanas estão impactando áreas naturais protegidas. Por isso, pesquisadores no México criaram o projeto VIGIA , que visa construir um sistema de vigilância autônoma de áreas protegidas. Um dos desafios propostos pela equipe foi criar um algoritmo que possa identificar um tipo específico de cacto em imagens aéreas. Eles disponibilizaram um dataset anotado no Kaggle, e é com ele que vamos trabalhar.
O dataset consta de imagens de áreas distintas e a anotação se tem presença ou não de cacto na imagem. Vamos utilizar a rede pré-treinada Xception. Primeiro vamos treinar a rede com as imagens RGB e depois vamos adicionar uma imagem tratada com o filtro NDVI visível.
Isso porque com esse índice podemos ter uma ideia da presença de vegetação e seu estado de saúde, pois há correlação entre atividade fotossintética das plantas e o teor de clorofila com as bandas vermelho e verde. No processo fotossintético a planta absorve a banda vermelha e a banda verde é refletida.m pré-processamento e aplicado na Aqui, a imagem original é concatenada com a imagem NDVI, quando o modo de treinamento é RGB_NDVI. E, com essas imagens com 6 canais, é treinada a rede neural.
A rede também é treinada utilizando 3 canais RGB, para comparar o desempenho.
