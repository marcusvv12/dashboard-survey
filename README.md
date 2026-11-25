# Dashboard: Clareza dos requisitos e retrabalho percebido

Parte 2 do trabalho de Medição e Experimentação em ES. É um dashboard em **Streamlit** que lê a **exportação bruta do formulário** e recalcula tudo a partir dela: descartes, índices, alfa e testes. Nenhum gráfico é imagem pronta.

## Rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

Para publicar, use o Streamlit Community Cloud: conecte o repositório **privado** do grupo e aponte para `app.py`. O protocolo prevê repositório privado, então não deixem a exportação bruta em repositório público.

## Dados

O dashboard já está configurado para a exportação oficial do Google Forms (`data/respostas_oficiais.csv`). Para atualizar com uma exportação nova, substituam o arquivo ou carreguem o CSV pela barra lateral.

Para usar outro formulário:

1. Exportem as respostas do formulário (CSV ou XLSX) e salvem em `data/`.
2. Em `config.yaml`:
   - `arquivo_dados`: caminho da exportação real;
   - `dados_simulados: false`;
   - `colunas`: o nome exato de cada coluna da exportação (copiem o cabeçalho);
   - `valores`: os rótulos usados no formulário (termo, elegibilidade, escalas, opções de E1);
   - `colunas_remover`: qualquer coluna com identificador (e-mail, nome…). Ela é descartada já na leitura;
   - `pre_teste.mediana_min`: o tempo mediano real do pré-teste, e depois `preenchido: true`.
3. Editem `desvios.md` com os desvios reais.
4. Quando a codificação de E2 estiver pronta, salvem `data/codificacao_E2.csv` (colunas `id, codificador_1, codificador_2, codigo_final`) e atualizem `arquivo_codificacao_e2`.

> **Critério (d), tempo de preenchimento:** o Google Forms **não registra a hora de início**. O Microsoft Forms registra ("Hora de início" e "Hora de conclusão"). Se a exportação não tiver início nem duração, o dashboard não aplica (d) e lança isso automaticamente como desvio.

Na apresentação, baixem a exportação do momento no painel do formulário e soltem o arquivo na barra lateral ("Carregar exportação bruta"). O dashboard recalcula tudo na hora.

## Páginas e o que a especificação pede

| Exigência | Onde está |
|---|---|
| Recebidas, descartadas e motivo de cada descarte | Visão geral e amostra |
| Likert em barras divergentes | Itens Likert |
| Teste, p-valor, tamanho de efeito e margem de erro por QP | QP1, QP2, QP3 |
| Filtros por segmento | Barra lateral: experiência, função, porte, forma de trabalho e canal |
| Alfa de Cronbach dos construtos | Confiabilidade (alfa) |
| Metodologia e desvios | Metodologia e desvios (`desvios.md` + desvios detectados nos dados) |
| Evidências (data/hora de cada resposta) | Evidências da coleta (gráficos por dia e CSV anonimizado) |

## Arquivos

- `app.py`: interface;
- `analise.py`: leitura, descartes e estatística, sem dependência do Streamlit (dá para conferir num notebook);
- `config.yaml`: tudo que muda de um formulário para outro;
- `desvios.md`: texto da página de desvios;
- `data/codificacao_E2.csv`: modelo para os dois codificadores da pergunta aberta;
- `evidencias/`: capturas datadas das divulgações, com nomes e fotos borrados.
