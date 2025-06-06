#!/usr/bin/env python3
"""
Rede Neural MLP para Jogo da Velha (Tic-Tac-Toe)

Implementação de uma rede neural Multi-Layer Perceptron (MLP) de 2 camadas
conforme especificações da imagem:

Topologia: 9 entradas + 1 bias → 9 neurônios ocultos → 9 saídas
- 9 entradas: posições do tabuleiro [x1, x2, ..., x9]
- Camada oculta: 9 neurônios com ativação tanh
- Camada de saída: 9 neurônios com ativação linear

Estrutura dos pesos (180 parâmetros):
- W1 (9×9): 81 pesos da entrada para camada oculta
- b1 (9):   9 bias da camada oculta  
- W2 (9×9): 81 pesos da camada oculta para saída
- b2 (9):   9 bias da camada de saída

Author: Baseado nas especificações da imagem fornecida
Date: 2025
"""

import numpy as np
from typing import List, Tuple


class RedeNeuralMLP:
    """
    Rede Neural MLP de 2 camadas para o jogo da velha.
    
    Arquitetura:
    - Entrada: 9 posições do tabuleiro
    - Camada oculta: 9 neurônios (tanh)
    - Camada saída: 9 neurônios (linear)
    """
    
    def __init__(self, pesos: np.ndarray = None):
        """
        Inicializa a rede neural.
        
        Args:
            pesos: Array com 180 pesos [W1, b1, W2, b2] ou None para pesos aleatórios
        """
        if pesos is not None:
            assert len(pesos) == 180, f"Esperado 180 pesos, recebido {len(pesos)}"
            self._extrair_pesos(pesos)
        else:
            self._inicializar_pesos_aleatorios()
    
    def _extrair_pesos(self, pesos: np.ndarray):
        """
        Extrai os pesos do vetor plano e os organiza nas matrizes.
        
        Layout dos pesos:
        - pesos[0:81]    → W1 (9×9)
        - pesos[81:90]   → b1 (9)  
        - pesos[90:171]  → W2 (9×9)
        - pesos[171:180] → b2 (9)
        """
        # Pontos de corte para extrair os pesos
        w1_fim = 81
        b1_fim = w1_fim + 9  # 90
        w2_fim = b1_fim + 81  # 171
        
        # Extrair e reformatar os pesos
        self.W1 = pesos[:w1_fim].reshape(9, 9)
        self.b1 = pesos[w1_fim:b1_fim]
        self.W2 = pesos[b1_fim:w2_fim].reshape(9, 9)
        self.b2 = pesos[w2_fim:]
        
        # Verificar dimensões
        assert self.W1.shape == (9, 9), f"W1 deve ser 9×9, obtido {self.W1.shape}"
        assert self.b1.shape == (9,), f"b1 deve ser (9,), obtido {self.b1.shape}"
        assert self.W2.shape == (9, 9), f"W2 deve ser 9×9, obtido {self.W2.shape}"
        assert self.b2.shape == (9,), f"b2 deve ser (9,), obtido {self.b2.shape}"
    
    def _inicializar_pesos_aleatorios(self):
        """Inicializa pesos aleatoriamente no intervalo [-1, 1]."""
        self.W1 = np.random.uniform(-1, 1, size=(9, 9))
        self.b1 = np.random.uniform(-1, 1, size=9)
        self.W2 = np.random.uniform(-1, 1, size=(9, 9))
        self.b2 = np.random.uniform(-1, 1, size=9)
    
    def propagacao(self, entrada: np.ndarray) -> np.ndarray:
        """
        Executa a propagação forward na rede.
        
        Args:
            entrada: Array com 9 elementos representando o tabuleiro
                    (valores: -1=O, 0=vazio, 1=X)
        
        Returns:
            saida: Array com 9 elementos (scores para cada posição)
        """
        assert len(entrada) == 9, f"Entrada deve ter 9 elementos, recebido {len(entrada)}"
        
        # Camada 1: Entrada → Oculta
        z1 = entrada @ self.W1 + self.b1  # Produto matricial + bias
        h1 = np.tanh(z1)                  # Ativação tanh
        
        # Camada 2: Oculta → Saída  
        z2 = h1 @ self.W2 + self.b2      # Produto matricial + bias
        saida = z2                        # Ativação linear (identidade)
        
        return saida
    
    def escolher_jogada(self, tabuleiro: np.ndarray) -> int:
        """
        Escolhe a melhor jogada baseada na saída da rede.
        
        Args:
            tabuleiro: Array com 9 elementos representando o estado do tabuleiro
        
        Returns:
            posicao: Índice da posição escolhida (0-8)
        """
        # Propagação para obter scores
        scores = self.propagacao(tabuleiro)
        
        # Mascarar posições ocupadas com -infinito
        scores_mascarados = np.where(tabuleiro == 0, scores, -np.inf)
        
        # Verificar se há jogadas válidas
        if np.all(np.isneginf(scores_mascarados)):
            # Fallback: escolher posição vazia aleatória
            posicoes_livres = [i for i, val in enumerate(tabuleiro) if val == 0]
            if posicoes_livres:
                return np.random.choice(posicoes_livres)
            else:
                return 0  # Último recurso
        
        # Retornar posição com maior score
        return int(np.argmax(scores_mascarados))
    
    def obter_pesos_planos(self) -> np.ndarray:
        """
        Converte os pesos da rede para um vetor plano.
        
        Returns:
            pesos: Array com 180 elementos [W1, b1, W2, b2]
        """
        return np.concatenate([
            self.W1.flatten(),  # 81 elementos
            self.b1,            # 9 elementos  
            self.W2.flatten(),  # 81 elementos
            self.b2             # 9 elementos
        ])
    
    def imprimir_arquitetura(self):
        """Imprime informações sobre a arquitetura da rede."""
        print("🧠 REDE NEURAL MLP - JOGO DA VELHA")
        print("=" * 50)
        print(f"Arquitetura: 9 → 9 → 9")
        print(f"Camada de entrada: 9 neurônios")
        print(f"Camada oculta: 9 neurônios (tanh)")  
        print(f"Camada de saída: 9 neurônios (linear)")
        print(f"Total de parâmetros: {self.contar_parametros()}")
        print("\nEstrutura dos pesos:")
        print(f"  W1: {self.W1.shape} = {self.W1.size} pesos")
        print(f"  b1: {self.b1.shape} = {self.b1.size} bias")
        print(f"  W2: {self.W2.shape} = {self.W2.size} pesos") 
        print(f"  b2: {self.b2.shape} = {self.b2.size} bias")
        print("=" * 50)
    
    def contar_parametros(self) -> int:
        """Conta o número total de parâmetros da rede."""
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size
    
    def estatisticas_pesos(self):
        """Imprime estatísticas dos pesos da rede."""
        todos_pesos = self.obter_pesos_planos()
        
        print("\n📊 ESTATÍSTICAS DOS PESOS:")
        print(f"   Mínimo: {todos_pesos.min():.4f}")
        print(f"   Máximo: {todos_pesos.max():.4f}")
        print(f"   Média: {todos_pesos.mean():.4f}")
        print(f"   Desvio padrão: {todos_pesos.std():.4f}")
        print(f"   Zeros (|w| < 0.001): {np.sum(np.abs(todos_pesos) < 0.001)}")


def exemplo_uso():
    """Demonstra o uso básico da rede neural."""
    print("🎮 EXEMPLO DE USO DA REDE NEURAL")
    print("=" * 50)
    
    # Criar rede com pesos aleatórios
    rede = RedeNeuralMLP()
    rede.imprimir_arquitetura()
    
    # Exemplo de tabuleiro (posições 0-8):
    # 0 | 1 | 2
    # ---------  
    # 3 | 4 | 5
    # ---------
    # 6 | 7 | 8
    
    tabuleiro_exemplo = np.array([
        1,  0, -1,    # X | _ | O
        0,  1,  0,    # _ | X | _  
        0, -1,  0     # _ | O | _
    ])
    
    print(f"\n🔢 Tabuleiro de exemplo:")
    print(f"   {tabuleiro_exemplo[:3]}")
    print(f"   {tabuleiro_exemplo[3:6]}")
    print(f"   {tabuleiro_exemplo[6:9]}")
    
    # Propagação
    scores = rede.propagacao(tabuleiro_exemplo)
    print(f"\n⚡ Scores da propagação:")
    print(f"   {scores[:3]}")
    print(f"   {scores[3:6]}")
    print(f"   {scores[6:9]}")
    
    # Escolha da jogada
    jogada = rede.escolher_jogada(tabuleiro_exemplo)
    print(f"\n🎯 Jogada escolhida: posição {jogada}")
    
    # Estatísticas dos pesos
    rede.estatisticas_pesos()


if __name__ == "__main__":
    exemplo_uso() 