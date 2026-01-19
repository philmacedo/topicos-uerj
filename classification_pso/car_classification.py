import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from torch.multiprocessing import Pool


# 1. Carregamento e Preprocessamento
#=============================================

# Carregar dataset Stanford Cars com cache
dataset = load_dataset("tanganke/stanford_cars")


# Obter lista de nomes das classes
label_names = dataset["train"].features["label"].names

# Dicionário índice -> nome completo do carro
id_to_name = {i: name for i, name in enumerate(label_names)}

# Lista dos tipos que queremos identificar
tipos_carro = [
    'Sedan', 'SUV', 'Coupe', 'Convertible', 'Hatchback',
    'Wagon', 'Minivan', 'Van', 'Crew Cab', 'Extended Cab',
    'Regular Cab', 'Roadster', 'Cargo Van', 'Pickup'
]

# Fixar a semente para reprodução
random.seed(42)

# Frações desejadas (exemplo: 30% dos dados)
frac_train = 0.3
frac_test = 0.3

train_subset = dataset["train"].select(random.sample(range(len(dataset["train"])), int(len(dataset["train"]) * frac_train)))
test_subset = dataset["train"].select(random.sample(range(len(dataset["train"])), int(len(dataset["train"]) * frac_train)))


# Função para extrair tipo do nome completo
def extrair_tipo_carro(nome_completo):
    for tipo in tipos_carro:
        if tipo in nome_completo:
            return tipo
    return 'Outro'

# Dicionário índice -> tipo do carro
id_to_type = {idx: extrair_tipo_carro(name) for idx, name in id_to_name.items()}

# Transformações para as imagens
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

class CarsDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset  # apenas referência, sem carregar tudo
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        tipo = id_to_type[label]
        if self.transform:
            image = self.transform(image)
        return image, label, tipo

# Datasets transformados
train_data = CarsDataset(train_subset, transform=transform)
val_data = CarsDataset(test_subset, transform=transform)


# DataLoaders
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=128, num_workers=0, pin_memory=True)



# --------------------
# Modelo - Cacheado
# --------------------
modelo_base = None  # variável global para guardar o modelo carregado

# Função que retorna uma cópia independente do modelo
def carregar_modelo():
    global modelo_base
    if modelo_base is None:
        print("Carregando modelo base...")
        modelo_base = AutoModelForImageClassification.from_pretrained("SriramSridhar78/sriram-car-classifier").to("cuda")
        for param in modelo_base.parameters():
            param.requires_grad = False
        for param in modelo_base.classifier.parameters():
            param.requires_grad = True   
    return modelo_base

#Cria uma nova instância do modelo com os pesos do modelo_base,
# para usar cópias independentes no PSO.
def clonar_modelo(device):
    base = carregar_modelo()
    config = base.config  # pega a configuração do modelo já carregado
    novo_modelo = AutoModelForImageClassification.from_config(config).to(device)
    novo_modelo.load_state_dict(base.state_dict())
    for param in novo_modelo.parameters():
        param.requires_grad = False
    for param in novo_modelo.classifier.parameters():
        param.requires_grad = True
    return novo_modelo

# --------------------
# PSO - Avaliação e Execução
# --------------------
def inicializar_particulas(n_particulas, limites_pos, limites_vel):
    particulas = []
    for _ in range(n_particulas):
        pos = np.array([np.random.uniform(low, high) for (low, high) in limites_pos])
        vel = np.array([np.random.uniform(low, high) for (low, high) in limites_vel])
        particulas.append({
            'pos': pos,
            'vel': vel,
            'melhor_pos': pos.copy(),
            'melhor_score': -np.inf
        })
    return particulas

def avaliar_particula(pos, train_loader, val_loader, device):
    lr = float(pos[0])

    print(f"\nAvaliando partícula - Learning Rate: {lr:.6f}")
    start_time = time.perf_counter()
    model = clonar_modelo(device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    model.train()
    for epoch in range(5):
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

    #validação
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acuracia = correct / total

    end_time = time.perf_counter()
    tempo = end_time - start_time
  
    print(f"Acurácia da partícula: {acuracia* 100:.2f}%")
    print(f"Tempo de avaliação da partícula: {tempo:.2f} segundos")
    return acuracia

def avaliar_wrapper(args):
    return avaliar_particula(*args)

def pso(n_particulas, n_iter, limites_pos, limites_vel, train_loader, val_loader, device, w=0.5, c1=1.5, c2=1.5):
    particulas = inicializar_particulas(n_particulas, limites_pos, limites_vel)
    gbest = {'pos': None, 'score': -np.inf}
    historico_scores = []
    historico_params = []


    for it in range(n_iter):
        #avalia as partículas em paralelo
        with Pool(processes=2) as pool:
            resultados = pool.map(avaliar_wrapper, [
                (p['pos'], train_loader, val_loader, device) for p in particulas
            ])
        # Atualiza o histórico de cada partícula com os resultados
        for i, score in enumerate(resultados):
            p = particulas[i]
            if score > p['melhor_score']:
                p['melhor_score'] = score
                p['melhor_pos'] = p['pos'].copy()
            if score > gbest['score']:
                gbest['score'] = score
                gbest['pos'] = p['pos'].copy()

        # Atualiza velocidades e posições
        for p in particulas:
            r1 = np.random.rand(len(p['pos']))
            r2 = np.random.rand(len(p['pos']))
            p['vel'] = (
                w * p['vel'] +
                c1 * r1 * (p['melhor_pos'] - p['pos']) +
                c2 * r2 * (gbest['pos'] - p['pos'])
            )
            p['pos'] += p['vel']
            for i, (low, high) in enumerate(limites_pos):
                p['pos'][i] = np.clip(p['pos'][i], low, high)

        historico_scores.append(gbest['score'])
        historico_params.append(gbest['pos'].copy())
        print(f"Iter {it+1}/{n_iter} - Melhor Score: {gbest['score']:.4f} - LR: {gbest['pos'][0]:.6f}")

    return gbest, historico_scores, historico_params

def plot_historico(historico_scores, historico_params):
    iteracoes = range(1, len(historico_scores) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(iteracoes, historico_scores, 'b-', label='Melhor Score (Acurácia)')
    plt.xlabel('Iteração')
    plt.ylabel('Acurácia')
    plt.title('Histórico de Acurácia no PSO')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(iteracoes, [p[0] for p in historico_params], 'r-', label='Learning Rate')
    plt.xlabel('Iteração')
    plt.ylabel('Learning Rate')
    plt.title('Histórico do Learning Rate no PSO')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    from torch.multiprocessing import freeze_support
    freeze_support()
    
    print("Início da execução...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")


    print("Inicializando PSO...")

    limites_pos = [(1e-5, 5e-3)]  # Apenas learning rate
    limites_vel = [(-1e-4, 1e-4)]

    melhor, historico_scores, historico_params = pso(n_particulas=5, n_iter=8, limites_pos=limites_pos, limites_vel=limites_vel,
                 train_loader=train_loader, val_loader=val_loader, device=device)

    print("Melhor solução encontrada:")
    print(f"Learning rate: {melhor['pos'][0]:.6f}, Acurácia: {melhor['score']* 100:.2f}%")

    plot_historico(historico_scores, historico_params)
