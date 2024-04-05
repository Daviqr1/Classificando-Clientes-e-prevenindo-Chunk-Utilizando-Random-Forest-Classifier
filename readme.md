# Desafio 2

Você trabalha em uma grande empresa de Cartão de Crédito e o diretor da empresa percebeu que o número de clientes que cancelam seus cartões tem aumentado significativamente, causando prejuízos enormes para a empresa.

O que fazer para evitar isso? Como saber as pessoas que têm maior tendência a cancelar o cartão?

A única informação que você possui é uma base de dados com informações dos clientes, tanto clientes atuais quanto os clientes que cancelaram o cartão.

# CONFIRA A RESOLUÇÃO
https://www.linkedin.com/posts/davi-rezende-09540b222_machinelearning-ciaeanciadedados-anaerlisededados-activity-7177799685196161026-7I0C?utm_source=share&utm_medium=member_desktop

# INTRODUÇÃO
Este estudo visa abordar a problemática do churn de clientes por meio da aplicação de técnicas de machine learning, com foco no algoritmo “RandomForestClassifier”. A escolha deste método se deve à sua capacidade de processar grandes volumes de dados e identificar padrões complexos que são imperceptíveis através de análises tradicionais. Ao classificar os clientes com base em sua propensão ao cancelamento, este trabalho busca não apenas prever comportamentos futuros, mas também fornecer uma base sólida para decisões estratégicas que visem a melhoria da satisfação e fidelização do cliente.

# METODOLOGIA
Preparação dos Dados: A análise iniciou-se com a preparação dos dados, uma etapa crítica para garantir a qualidade e a precisão do modelo de machine learning. Utilizamos a biblioteca Pandas para limpar e formatar o conjunto de dados, tratando valores ausentes, removendo duplicatas e convertendo tipos de dados quando necessário. A coluna “Inatividade 12m” foi essencial para a categorização inicial dos clientes, permitindo uma segmentação baseada no comportamento de uso do cartão de crédito ao longo dos últimos 12 meses.

# SELEÇÃO DE PARÂMETROS
Os parâmetros selecionados para o treinamento do modelo incluem ‘Idade’, ‘Dependentes’, ‘Meses como Cliente’, ‘Produtos Contratados’, ‘Inatividade 12m’, ‘Contatos 12m’, ‘Limite’, ‘Limite Consumido’, ‘Limite Disponível’, ‘Mudanças Transacoes_Q4_Q1’, ‘Valor Transacoes 12m’, ‘Qtde Transacoes 12m’, ‘Mudança Qtde Transações_Q4_Q1’ e ‘Taxa de Utilização Cartão’. Esses parâmetros foram escolhidos por sua relevância na representação do perfil financeiro e comportamental dos clientes, fornecendo uma base sólida para a previsão de cancelamentos.

# CLASSIFICAÇÃO DESEJADA
Buscamos classificar os clientes em três categorias distintas: ‘Ativo’, ‘Observação’ e ‘Crítico’. A classificação ‘Ativo’ indica clientes com uso regular do cartão, ‘Observação’ aponta para aqueles com sinais de inatividade e ‘Crítico’ identifica clientes com alta probabilidade de cancelamento. Esta classificação permite uma intervenção direcionada para cada segmento de clientes.

# MODELAGEM MATEMÁTICA
O RandomForestClassifier é um algoritmo de ensemble que combina múltiplas árvores de decisão para produzir uma previsão mais estável e precisa. Cada árvore é construída a partir de uma amostra aleatória do conjunto de dados, e a decisão final é tomada pela agregação das previsões das árvores individuais - o que reduz o risco de overfitting. Matematicamente, o modelo utiliza medidas de impureza como o índice Gini ou a entropia para otimizar a seleção de divisões em cada nó das árvores. A importância das variáveis é determinada pela redução média da impureza que cada variável proporciona nas árvores do modelo.

# RESULTADOS DA DISCUSSÃO
Análise de Desempenho:
O modelo “RandomForestClassifier” alcançou uma acurácia notável de 99,80% na tarefa de classificação dos clientes. A matriz de confusão gerada foi:

![image](https://github.com/Daviqr1/Classificando-Clientes-e-prevenindo-Chunk-Utilizando-Random-Forest-Classifier/assets/84293017/8b2908b8-57f1-495e-8d84-196f4065accf)

# INTERPRETAÇÃO DO GRÁFICO
A Matriz de Confusão é uma ferramenta visual importante para avaliar o desempenho de modelos de classificação. Ela apresenta as contagens de previsões corretas e incorretas feitas pelo modelo, categorizadas pelas classes reais e previstas. Aqui está uma explicação do gráfico:

* Classe 0 (Ativo): O modelo previu corretamente 63 instâncias como ‘Ativo’, e não houve previsões incorretas para esta classe.
* Classe 1 (Observação): Não há instâncias previstas ou reais para a classe ‘Observação’, indicando que esta classe pode não estar presente no conjunto de dados ou o modelo não identificou nenhuma instância desta classe.
* Classe 2 (Crítico): O modelo previu corretamente 948 instâncias como ‘Crítico’. Houve 2 casos onde o modelo previu ‘Observação’ quando na realidade eram ‘Crítico’, representando falsos negativos.
  
A ausência de previsões incorretas para as classes ‘Ativo’ e ‘Crítico’ sugere que o modelo tem uma alta precisão na identificação dessas categorias. Os falsos negativos para a classe ‘Crítico’ são áreas que podem requerer atenção para melhorar o modelo. A matriz indica um desempenho excepcional do modelo, especialmente na identificação de clientes ‘Críticos’, que é essencial para a prevenção do cancelamento de cartões de crédito.

# CONFIABILIDADE DO MODELO
Para assegurar a máxima confiabilidade das previsões, o modelo foi treinado utilizando um conjunto de dados diversificado e representativo, abrangendo uma ampla gama de comportamentos de clientes. Os dados necessários para alcançar tal confiabilidade incluem, mas não se limitam a, informações demográficas, histórico de transações, padrões de uso do cartão, interações com o serviço ao cliente e feedbacks recebidos. A inclusão de variáveis temporais, como ‘Mudanças Transacoes_Q4_Q1’, permite ao modelo capturar tendências e mudanças no comportamento do cliente ao longo do tempo.

# IMPORTÃNCIA DAS VARIÁVEIS NO MODELO PREDITIVO
A análise de importância das variáveis é um aspecto crucial na compreensão do modelo “RandomForestClassifier”. Nossa investigação revelou que as variáveis ‘Inatividade 12m’, ‘Limite Consumido’ e ‘Taxa de Utilização do Cartão’ emergiram como os principais indicadores de cancelamento. Essa descoberta é intuitiva, pois clientes menos engajados, que utilizam uma menor porção do seu limite de crédito, tendem a demonstrar um comportamento de desvinculação com os serviços do banco.
Visualização Gráfica da Importância das Variáveis: A visualização gráfica é uma ferramenta poderosa para ilustrar a relevância das variáveis. No gráfico que acompanha este texto, podemos observar a distribuição da importância das variáveis. A função lambda utilizada para classificar a tabela ‘clientes1’ inicialmente serve como um ponto de comparação para a importância atribuída pelo modelo. O gráfico destaca como o algoritmo treinado prioriza as mesmas variáveis que o sistema de classificação real do banco, corroborando a alta acurácia do modelo.

![image](https://github.com/Daviqr1/Classificando-Clientes-e-prevenindo-Chunk-Utilizando-Random-Forest-Classifier/assets/84293017/1a249397-e1af-4cf0-aa76-250241ec2abb)

# INTERPRETAÇÃO DOS RESULTADOS GRÁFICOS
Ao examinar o gráfico, notamos que a variável ‘Inatividade 12m’ possui um peso significativo na decisão do modelo, seguida pelo ‘Limite Consumido’ e pela ‘Taxa de Utilização do Cartão’. Essa hierarquia na importância das variáveis reflete diretamente na capacidade do modelo de prever com precisão o cancelamento dos serviços, como evidenciado pela acurácia elevada.

# IMPLICAÇÕES PRÁTICAS
Os resultados obtidos pelo modelo oferecem insights valiosos para a formulação de estratégias de retenção. Ao identificar os clientes com maior risco de churn, a empresa pode direcionar esforços de marketing e ofertas personalizadas para aumentar a satisfação e a fidelização.

# LIMITAÇÕES E FUTURAS DIREÇÕES
Embora o modelo tenha demonstrado alta acurácia, é importante reconhecer as limitações inerentes a qualquer estudo analítico. A performance do modelo pode variar com a introdução de novos dados ou mudanças no comportamento do mercado. Estudos futuros podem explorar a integração de dados de mídias sociais e análise de sentimentos para enriquecer o conjunto de dados e refinar ainda mais as previsões do modelo.


# REFERÊNCIAS
* Documentação da biblioteca Pandas
* Documentação da biblioteca Scikit-learn
* Artigos relevantes sobre técnicas de machine learning e análise de dados
* Dados Públicos da tabela Cliente.csv 

























