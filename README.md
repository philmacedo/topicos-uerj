# 🚗 Classificação de Carros com Otimização via PSO (Particle Swarm Optimization)

> Um projeto de Visão Computacional que utiliza Transfer Learning e Inteligência de Enxames para otimizar a classificação de veículos.

## 📌 Sobre o Projeto
Este projeto visa resolver o problema da **classificação de tipos de carros** (Sedan, SUV, Pickup, etc.) a partir de imagens. A identificação automatizada de veículos é crucial para monitoramento urbano, controle de tráfego e gestão de frotas.

A principal inovação deste trabalho é a utilização do algoritmo **PSO (Particle Swarm Optimization)** para encontrar automaticamente a melhor **Taxa de Aprendizado (Learning Rate)**, maximizando a acurácia do modelo.

## 🎯 Objetivos
1.  Utilizar um modelo pré-treinado como extrator de características (**Transfer Learning**).
2.  Agrupar as 196 classes originais do dataset em **14 categorias funcionais**.
3.  Implementar o algoritmo PSO para otimizar os hiperparâmetros (Learning Rate) da rede neural.

---

## 🛠️ Tecnologias e Dataset

* **Linguagem:** Python
* **Frameworks:** PyTorch, Transformers (Hugging Face), Torchvision.
* **Dataset:** [Stanford Cars (tanganke/stanford_cars)](https://huggingface.co/datasets/tanganke/stanford_cars)
* **Modelo Base:** `SriramSridhar78/sriram-car-classifier`
* **Técnica de Otimização:** Particle Swarm Optimization (PSO).

---

## ⚙️ Metodologia

### 1. Pré-processamento
O dataset original possui rótulos muito específicos (ex: *Marca + Modelo + Ano*). Foi criado um mapeamento para generalizar esses rótulos em **14 classes**:
`Sedan`, `SUV`, `Coupe`, `Convertible`, `Hatchback`, `Wagon`, `Minivan`, `Van`, `Crew Cab`, `Extended Cab`, `Regular Cab`, `Roadster`, `Cargo Van`, `Pickup`.

### 2. Engenharia do Modelo
* Congelamento das camadas convolucionais do modelo pré-treinado.
* Treinamento apenas da última camada de classificação (Fine-tuning).
* Uso de paralelismo na GPU para treinar múltiplas partículas simultaneamente.

### 3. Otimização PSO
Cada "partícula" do enxame representa um valor de *Learning Rate*. O algoritmo ajusta esses valores baseado na experiência individual (melhor acurácia da partícula) e coletiva (melhor acurácia do enxame).

---

## 📊 Resultados Alcançados

O algoritmo foi configurado com os seguintes parâmetros finais:
* **Partículas:** 5
* **Iterações:** 8
* **Épocas de treino por partícula:** 5
* **Limite de posição (Search Space):** [1e-5, 5e-3]

### Desempenho
* 🏆 **Melhor Acurácia:** **92.43%**
* ⚡ **Melhor Learning Rate encontrado:** `0.003949`

### Gráficos de Convergência

#### Histórico do Learning Rate
*O algoritmo convergiu rapidamente para valores próximos a 0.004 e se estabilizou.*
![Gráfico de Learning Rate](./assets/historico_lr_92_5epoch.png)
*(Certifique-se de que a imagem está no repositório com esse nome)*

#### Histórico de Acurácia
*Houve um salto de desempenho na iteração 5, atingindo o pico de 92.43%.*
![Gráfico de Acurácia](./assets/historico_acuracia_92_5epoch.png)
*(Certifique-se de que a imagem está no repositório com esse nome)*

---

## 💻 Exemplo de Código (Core do PSO)

Abaixo, um trecho da implementação do algoritmo PSO utilizado para atualizar as velocidades e posições das partículas:

```python
# Trecho da função pso()
for p in particulas:
    r1 = np.random.rand(len(p['pos']))
    r2 = np.random.rand(len(p['pos']))
    
    # Atualização da velocidade (Inércia + Cognitivo + Social)
    p['vel'] = (
        w * p['vel'] +
        c1 * r1 * (p['melhor_pos'] - p['pos']) +
        c2 * r2 * (gbest['pos'] - p['pos'])
    )
    
    # Atualização da posição (Novo Learning Rate)
    p['pos'] += p['vel']
    
    # Clip para manter dentro dos limites definidos
    for i, (low, high) in enumerate(limites_pos):
        p['pos'][i] = np.clip(p['pos'][i], low, high)
```

## 👨‍💻 Autor

**Philipy Macêdo** -> Engenharia de Sistemas e Computação - UERJ
