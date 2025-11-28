<p align="center">
  <img src="https://raw.githubusercontent.com/monostate/aitraining/main/docs/images/terminal-wizard.png" alt="AITraining Interactive Wizard" width="700">
</p>

<p align="center">
  <a href="https://pypi.org/project/aitraining/"><img src="https://img.shields.io/pypi/v/aitraining.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/aitraining/"><img src="https://img.shields.io/pypi/pyversions/aitraining.svg" alt="Python versions"></a>
  <a href="https://github.com/monostate/aitraining/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <b>Treine modelos de ML de ponta com codigo minimo</b>
</p>

<p align="center">
  <a href="README.md">English</a> | Portugues
</p>

---

AITraining e uma plataforma avancada de treinamento de machine learning construida sobre o [AutoTrain Advanced](https://github.com/huggingface/autotrain-advanced). Oferece uma interface simplificada para fine-tuning de LLMs, modelos de visao e mais.

## Destaques

### Conversao Automatica de Dataset

Alimente qualquer formato de dataset e o AITraining detecta e converte automaticamente. Suporta 6 formatos de entrada com deteccao automatica:

| Formato | Deteccao | Colunas de Exemplo |
|---------|----------|-------------------|
| **Alpaca** | instruction/input/output | `{"instruction": "...", "output": "..."}` |
| **ShareGPT** | pares from/value | `{"conversations": [{"from": "human", ...}]}` |
| **Messages** | role/content | `{"messages": [{"role": "user", ...}]}` |
| **Q&A** | variantes question/answer | `{"question": "...", "answer": "..."}` |
| **DPO** | prompt/chosen/rejected | Para treinamento de preferencia |
| **Plain Text** | Coluna de texto unica | Texto bruto para pretraining |

```bash
aitraining llm --train --auto-convert-dataset --chat-template gemma3 \
  --data-path tatsu-lab/alpaca --model google/gemma-3-270m-it
```

### 32 Chat Templates

Biblioteca completa de templates com controle de peso por token:

- **Familia Llama**: llama, llama-3, llama-3.1
- **Familia Gemma**: gemma, gemma-2, gemma-3, gemma-3n
- **Outros**: mistral, qwen-2.5, phi-3, phi-4, chatml, alpaca, vicuna, zephyr

```python
from autotrain.rendering import get_renderer, ChatFormat, RenderConfig

config = RenderConfig(format=ChatFormat.CHATML, only_assistant=True)
renderer = get_renderer('chatml', tokenizer, config)
encoded = renderer.build_supervised_example(conversation)
# Retorna: {'input_ids', 'labels', 'token_weights', 'attention_mask'}
```

### Ambientes RL Personalizados

Construa funcoes de recompensa personalizadas para treinamento PPO com tres tipos de ambiente:

```bash
# Geracao de texto com recompensa customizada
aitraining llm --train --trainer ppo \
  --rl-env-type text_generation \
  --rl-env-config '{"stop_sequences": ["</s>"]}' \
  --rl-reward-model-path ./reward_model

# Recompensas multi-objetivo (corretude + formatacao)
aitraining llm --train --trainer ppo \
  --rl-env-type multi_objective \
  --rl-env-config '{"reward_components": {"correctness": {"type": "keyword"}, "formatting": {"type": "length"}}}' \
  --rl-reward-weights '{"correctness": 1.0, "formatting": 0.1}'
```

### Sweeps de Hiperparametros

Otimizacao automatizada com Optuna, busca aleatoria ou grid search:

```python
from autotrain.utils import HyperparameterSweep, SweepConfig, ParameterRange

config = SweepConfig(
    backend="optuna",
    optimization_metric="eval_loss",
    optimization_mode="minimize",
    num_trials=20,
)

sweep = HyperparameterSweep(
    objective_function=train_model,
    config=config,
    parameters=[
        ParameterRange("learning_rate", "log_uniform", low=1e-5, high=1e-3),
        ParameterRange("batch_size", "categorical", choices=[4, 8, 16]),
    ]
)
result = sweep.run()
# Retorna best_params, best_value, historico de trials
```

### Metricas de Avaliacao Aprimoradas

8 metricas alem da loss, com callbacks para avaliacao periodica:

| Metrica | Tipo | Caso de Uso |
|---------|------|-------------|
| **Perplexity** | Auto-computada | Qualidade do modelo de linguagem |
| **BLEU** | Geracao | Traducao, sumarizacao |
| **ROUGE** (1/2/L) | Geracao | Sumarizacao |
| **BERTScore** | Geracao | Similaridade semantica |
| **METEOR** | Geracao | Traducao |
| **F1/Accuracy** | Classificacao | Metricas padrao |
| **Exact Match** | QA | Question answering |

```python
from autotrain.evaluation import Evaluator, EvaluationConfig, MetricType

config = EvaluationConfig(
    metrics=[MetricType.PERPLEXITY, MetricType.BLEU, MetricType.ROUGE, MetricType.BERTSCORE],
    save_predictions=True,
)
evaluator = Evaluator(model, tokenizer, config)
result = evaluator.evaluate(dataset)
```

### Merge Automatico de LoRA

Apos treinamento PEFT, automaticamente faz merge dos adapters e salva modelos prontos para deploy:

```bash
# Padrao: faz merge dos adapters no modelo completo
aitraining llm --train --peft --model meta-llama/Llama-3.2-1B

# Manter adapters separados (arquivos menores)
aitraining llm --train --peft --no-merge-adapter --model meta-llama/Llama-3.2-1B
```

---

## Capturas de Tela

<p align="center">
  <img src="https://raw.githubusercontent.com/monostate/aitraining/main/docs/images/chat-screenshot.png" alt="Interface de chat para testar modelos treinados" width="700">
  <br>
  <em>Interface de chat integrada para testar modelos treinados com historico de conversas</em>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/monostate/aitraining/main/docs/images/tui-wandb.png" alt="TUI com integracao W&B LEET" width="700">
  <br>
  <em>TUI com visualizacao de metricas em tempo real via W&B LEET</em>
</p>

---

## Instalacao

```bash
pip install aitraining
```

Requisitos: Python >= 3.10, PyTorch

## Inicio Rapido

### Assistente Interativo

```bash
aitraining
```

O assistente guia voce atraves de:
1. Selecao do tipo de trainer (LLM, visao, NLP, tabular)
2. Selecao de modelo com catalogos curados do HuggingFace
3. Configuracao de dataset com deteccao automatica de formato
4. Parametros avancados (PEFT, quantizacao, sweeps)

### Arquivo de Configuracao

```bash
aitraining --config config.yaml
```

### API Python

```python
from autotrain.trainers.clm import train
from autotrain.trainers.clm.params import LLMTrainingParams

config = LLMTrainingParams(
    model="meta-llama/Llama-3.2-1B",
    data_path="seu-dataset",
    trainer="sft",
    epochs=3,
    batch_size=4,
    lr=2e-5,
    peft=True,
    auto_convert_dataset=True,
    chat_template="llama3",
)

train(config)
```

---

## Comparacao

### AITraining vs AutoTrain vs Tinker

| Recurso | AutoTrain | AITraining | Tinker |
|---------|-----------|------------|--------|
| **Trainers** |
| SFT/DPO/ORPO | Sim | Sim | Sim |
| PPO (RLHF) | Basico | Aprimorado (TRL) | Avancado |
| Reward Modeling | Sim | Sim | Nao |
| Destilacao de Conhecimento | Nao | Sim (KL + CE loss) | Sim (texto-only) |
| **Dados** |
| Deteccao Auto de Formato | Nao | Sim (6 formatos) | Nao |
| Biblioteca de Chat Templates | Basica | 32 templates | 5 templates |
| Mapeamento de Colunas Runtime | Nao | Sim | Nao |
| Extensao de Conversas | Nao | Sim | Nao |
| **Treinamento** |
| Sweeps de Hiperparametros | Nao | Sim (Optuna) | Manual |
| Ambientes RL Personalizados | Nao | Sim (3 tipos) | Sim |
| Recompensas Multi-objetivo | Nao | Sim | Sim |
| Pipeline Forward-Backward | Nao | Sim | Sim |
| RL Async Off-Policy | Nao | Nao | Sim |
| Stream Minibatch | Nao | Nao | Sim |
| **Avaliacao** |
| Metricas Alem da Loss | Nao | 8 metricas | Manual |
| Callbacks de Eval Periodico | Nao | Sim | Sim |
| Registro de Metricas Custom | Nao | Sim | Nao |
| **Interface** |
| Wizard CLI Interativo | Nao | Sim | Nao |
| TUI (Experimental) | Nao | Sim | Nao |
| Visualizador W&B LEET | Nao | Sim | Sim |
| **Hardware** |
| Apple Silicon (MPS) | Limitado | Completo | Nao |
| Quantizacao (int4/int8) | Sim | Sim | Desconhecido |
| Multi-GPU | Sim | Sim | Sim |
| **Cobertura de Tarefas** |
| Tarefas de Visao | Sim | Sim | Nao |
| Tarefas NLP | Sim | Sim | Nao |
| Tarefas Tabulares | Sim | Sim | Nao |
| Ambientes Tool Use | Nao | Nao | Sim |
| RL Multiplayer | Nao | Nao | Sim |

---

## Tarefas Suportadas

| Tarefa | Trainers | Status |
|--------|----------|--------|
| Fine-tuning de LLM | SFT, DPO, ORPO, PPO, Reward, Destilacao | Estavel |
| Classificacao de Texto | Single/Multi-label | Estavel |
| Classificacao de Tokens | NER, POS tagging | Estavel |
| Sequence-to-Sequence | Traducao, Sumarizacao | Estavel |
| Classificacao de Imagens | Single/Multi-label | Estavel |
| Deteccao de Objetos | YOLO, DETR | Estavel |
| Treinamento VLM | Vision-Language Models | Beta |
| Tabular | XGBoost, sklearn | Estavel |
| Sentence Transformers | Similaridade semantica | Estavel |
| QA Extrativo | Formato SQuAD | Estavel |

---

## Exemplo de Configuracao

```yaml
task: llm-sft
base_model: meta-llama/Llama-3.2-1B
project_name: meu-finetune

data:
  path: seu-dataset
  train_split: train
  auto_convert_dataset: true
  chat_template: llama3

params:
  epochs: 3
  batch_size: 4
  lr: 2e-5
  peft: true
  lora_r: 16
  lora_alpha: 32
  quantization: int4
  mixed_precision: bf16

# Opcional: sweep de hiperparametros
sweep:
  enabled: true
  backend: optuna
  n_trials: 10
  metric: eval_loss
```

---

## Documentacao

- [Guia do Assistente Interativo](docs/interactive_wizard.md)
- [Formatos de Dataset & Conversao](docs/dataset_formats.md)
- [Referencia de Trainers](docs/trainers/README.md)
- [API Python](docs/api/PYTHON_API.md)
- [Referencia da API RL](docs/reference/RL_API_REFERENCE.md)

---

## Licenca

Apache 2.0 - Veja [LICENSE](LICENSE) para detalhes.

Baseado no [AutoTrain Advanced](https://github.com/huggingface/autotrain-advanced) da Hugging Face.

---

<p align="center">
  <a href="https://monostate.ai">Monostate AI</a>
</p>
