# Rede Neural MLP para Jogo da Velha

## 📋 Visão Geral

Implementação de uma **Rede Neural MLP (Multi-Layer Perceptron) de 2 camadas** para jogar o jogo da velha, integrada com **Algoritmo Genético** conforme especificações fornecidas.

## 🏗️ Arquitetura da Rede

```
9 entradas → 9 neurônios ocultos → 9 saídas
    ↓              ↓                ↓
  x1-x9         tanh()           linear
(tabuleiro)   (ativação)        (scores)
```

### Detalhes Técnicos:
- **Entrada**: 9 neurônios (posições do tabuleiro: -1=O, 0=vazio, 1=X)
- **Camada Oculta**: 9 neurônios com ativação `tanh`
- **Camada Saída**: 9 neurônios com ativação linear
- **Total de Parâmetros**: 180 pesos

### Estrutura dos Pesos:
```
W1 (9×9):  81 pesos  (entrada → oculta)
b1 (9):     9 bias   (camada oculta)
W2 (9×9):  81 pesos  (oculta → saída)  
b2 (9):     9 bias   (camada saída)
-------------------------------
Total:    180 pesos
```

## 🧬 Integração com Algoritmo Genético

### Matriz da População
Cada linha da matriz representa um **indivíduo** (rede neural):

```
┌─────────────────────────────────┬─────┐
│           0 … 179               │ 180 │
├─────────────────────────────────┼─────┤
│ -0.1 | -0.008 | -0.5 | 0.0625… │ apt │
│ 0.23 | -0.456 | 0.78 | -0.123… │ apt │
│ …                               │ …   │
└─────────────────────────────────┴─────┘
```

- **Colunas 0-179**: 180 pesos da rede neural
- **Coluna 180**: Função aptidão (quanto menor, melhor)
- **Valores**: Pesos aleatórios entre -1 e 1

## 📁 Arquivos Implementados

### 1. `src/neural_network.py`
Implementação base da rede neural MLP:
- Classe `RedeNeuralMLP`
- Propagação forward
- Exemplo de uso

### 2. `src/rede_neural_genetica.py`
Integração com algoritmo genético:
- Classe `RedeNeural` (versão otimizada)
- Classe `PopulacaoGenetica`
- Gerenciamento da matriz de população

### 3. `src/exemplo_populacao.py`
Demonstração completa:
- Estrutura da matriz conforme imagem
- Teste de propagação
- Exemplos práticos

## 🚀 Como Usar

### Exemplo Básico
```python
from rede_neural_genetica import PopulacaoGenetica, RedeNeural

# Criar população genética
populacao = PopulacaoGenetica(tamanho_populacao=50)

# Usar um indivíduo como rede neural
rede = populacao.criar_rede_neural(0)

# Testar com tabuleiro do jogo da velha
tabuleiro = np.array([1, 0, -1, 0, 1, 0, 0, -1, 0])
jogada = rede.escolher_jogada(tabuleiro)
print(f"Jogada escolhida: posição {jogada}")
```

### Propagação Manual
```python
# Propagação forward
scores = rede.propagacao(tabuleiro)
print(f"Scores para cada posição: {scores}")
```

## 🔍 Fórmulas da Propagação

### Camada 1 (Entrada → Oculta)
```
z1 = X @ W1 + b1
h1 = tanh(z1)
```

### Camada 2 (Oculta → Saída)
```
z2 = h1 @ W2 + b2
y = z2  (ativação linear)
```

Onde:
- `X`: Entrada (9 posições do tabuleiro)
- `W1, W2`: Matrizes de pesos
- `b1, b2`: Vetores de bias
- `@`: Produto matricial

## 🎯 Escolha da Jogada

A rede escolhe a jogada através do **argmax** das saídas, considerando apenas posições livres:

```python
# Mascarar posições ocupadas
scores_masked = np.where(tabuleiro == 0, scores, -inf)

# Escolher posição com maior score
jogada = np.argmax(scores_masked)
```

## 📊 Características da Implementação

✅ **Conformidade com Especificações**:
- ✓ Topologia 9×9 com 180 pesos
- ✓ Propagação forward implementada
- ✓ Integração com matriz do algoritmo genético
- ✓ Valores aleatórios entre -1 e 1
- ✓ Estrutura 181 colunas (180 pesos + 1 aptidão)

✅ **Funcionalidades Extras**:
- ✓ Classe para gerenciar população genética
- ✓ Métodos para extrair estatísticas
- ✓ Exemplos de uso detalhados
- ✓ Testes com tabuleiros do jogo da velha

## 🧪 Testando a Implementação

```bash
# Teste básico da rede neural
python src/neural_network.py

# Teste da integração genética
python src/rede_neural_genetica.py

# Demonstração completa
python src/exemplo_populacao.py
```

## 📈 Próximos Passos

Para completar o sistema, seria necessário implementar:

1. **Função de Aptidão**: Avaliação através de jogos contra oponentes
2. **Operadores Genéticos**: Seleção, crossover e mutação
3. **Loop Principal**: Evolução da população ao longo das gerações

A implementação atual fornece a **base sólida** da rede neural conforme especificado, pronta para integração com o algoritmo genético completo.

---

**Autor**: Implementação baseada nas especificações fornecidas  
**Data**: 2025 