# Projeto Integrado

## 1. Contextualização Geral

O projeto "Auto Simula?", desenvolvido originalmente como parte do Projeto Interdisciplinar 4 do curso de Sistemas de Informação, teve como objetivo criar uma plataforma interativa capaz de auxiliar usuários a tomar decisões financeiras relacionadas à compra ou troca de veículos. A solução combina raspagem de dados reais do mercado automotivo com simulações financeiras personalizadas, oferecendo recomendações práticas com base no perfil econômico do usuário.

A proposta surgiu da necessidade de disponibilizar ao público uma ferramenta confiável e educativa, capaz de reduzir riscos financeiros e promover escolhas mais conscientes ao lidar com produtos de alto impacto econômico, como automóveis.

Já o projeto atual, embora baseado em uma base de dados mockada e utilizado para fins exclusivamente técnicos e didáticos, mantém relação direta com a proposta original. Ele representa uma etapa fundamental para a evolução técnica da plataforma, pois explora como algoritmos de Machine Learning podem complementar ou aprimorar funcionalidades relacionadas a previsão, categorização e agrupamento de veículos.

Assim, este documento descreve o projeto original e esclarece como o projeto atual se conecta a ele em termos de objetivos, fundamentos e aplicações práticas.

## 2. O Projeto Original – "Auto Simula?"

### Objetivo Geral

O projeto original buscou desenvolver um site interativo que:

- Coleta informações financeiras do usuário e dados sobre seu veículo atual
- Cruza essas informações com dados reais raspados do Carflix
- Realiza simulações de financiamento e custo total do veículo
- Recomenda alternativas viáveis dentro do perfil econômico analisado
- Indica se vale a pena ou não realizar a troca do carro
- Apresenta estimativas de parcelas, entrada necessária, custo-benefício e economia potencial

A plataforma se posiciona como uma ferramenta de apoio à decisão, acessível, clara e baseada em dados reais.

### Motivação e Relevância

O projeto justificou-se por três pilares centrais:

#### Conscientização Financeira
Permitir que pessoas entendam melhor o impacto econômico da compra de um veículo.

#### Transparência de Mercado
Facilitar o acesso a informações de preços, tendências e oportunidades.

#### Desenvolvimento Sustentável
Dialoga com:
- ODS 4 (Educação de Qualidade)
- ODS 8 (Crescimento Econômico)
- ODS 12 (Consumo Responsável)

Além disso, o projeto apresenta forte potencial de escala e monetização futura, por meio de:
- Parcerias com concessionárias
- Anúncios segmentados
- Serviços premium

### Funcionalidades Estruturantes

O projeto original foi organizado com base em:

- Questionário financeiro
- Coleta de dados reais via Web Scraping
- Simulação de financiamento
- Cálculo de juros, entrada e parcelas
- Recomendação personalizada de veículos
- Análise de custo-benefício e economia
- Comparativo visual entre opções
- Planejamento de MVP, backlog e roadmap futuro

Embora não tenha incluído IA em sua versão inicial (além de processamentos tradicionais), o escopo abriu espaço para uma futura evolução com algoritmos inteligentes.

## 3. O Projeto Atual – Base Mockada e Modelos de Machine Learning

O projeto atual utiliza uma base de dados simulada (mockada) com atributos automotivos fictícios — como preço, quilometragem, câmbio, combustível, entre outros — para permitir experimentação e estudo de técnicas de Machine Learning.

### Algoritmos Aplicados

A proposta prática consiste em aplicar diferentes algoritmos:

#### Árvore de Decisão (Regressão)
Para prever preços de veículos com base em atributos relevantes.

#### KNN (Classificação)
Para prever categorias, como tipo de câmbio, a partir de características numéricas.

#### K-Means (Clusterização)
Para agrupar veículos com perfis semelhantes e identificar padrões naturais.

### Observação Importante

Ainda que o dataset seja mockado e não possua conexão direta com dados reais do Carflix, ele simula estruturalmente o tipo de informação utilizada no projeto original, permitindo testes e validações em um ambiente controlado.