#!/usr/bin/env python3
"""
Rede Neural MLP integrada com Algoritmo Genético para Jogo da Velha

Implementação seguindo as especificações:
- Cada linha da população representa um indivíduo
- Colunas 0-179: 180 pesos da rede neural
- Coluna 180: função aptidão
- População inicial com valores aleatórios entre -1 e 1

Estrutura dos pesos (180 parâmetros por indivíduo):
- W1 (9×9): 81 pesos da entrada para camada oculta
- b1 (9):   9 bias da camada oculta  
- W2 (9×9): 81 pesos da camada oculta para saída
- b2 (9):   9 bias da camada de saída

Author: Baseado nas especificações fornecidas
Date: 2025
"""

import numpy as np
from typing import List, Tuple, Optional


class RedeNeural:
    """
    Rede Neural MLP de 2 camadas para o jogo da velha.
    
    Arquitetura: 9 → 9 → 9
    - Entrada: 9 posições do tabuleiro
    - Camada oculta: 9 neurônios (tanh)
    - Camada saída: 9 neurônios (linear)
    """
    
    def __init__(self, pesos: np.ndarray):
        """
        Inicializa a rede neural com pesos específicos.
        
        Args:
            pesos: Array com 180 pesos [W1, b1, W2, b2]
        """
        assert len(pesos) == 180, f"Esperado 180 pesos, recebido {len(pesos)}"
        self._extrair_pesos(pesos)
    
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
        b1_fim = w1_fim + 9      # 90
        w2_fim = b1_fim + 81     # 171
        
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


class PopulacaoGenetica:
    """
    Classe para gerenciar a população do algoritmo genético.
    
    Cada linha da matriz representa um indivíduo:
    - Colunas 0-179: 180 pesos da rede neural
    - Coluna 180: função aptidão
    """
    
    def __init__(self, tamanho_populacao: int = 50):
        """
        Inicializa a população genética.
        
        Args:
            tamanho_populacao: Número de indivíduos na população
        """
        self.tamanho_populacao = tamanho_populacao
        self.num_pesos = 180
        self.num_colunas = 181  # 180 pesos + 1 aptidão
        
        # Inicializar população com valores aleatórios
        self.populacao = self._criar_populacao_inicial()
    
    def _criar_populacao_inicial(self) -> np.ndarray:
        """
        Cria população inicial com pesos aleatórios entre -1 e 1.
        
        Returns:
            populacao: Matriz (tamanho_populacao × 181)
        """
        # Criar matriz da população
        populacao = np.zeros((self.tamanho_populacao, self.num_colunas))
        
        # Preencher colunas 0-179 com pesos aleatórios entre -1 e 1
        populacao[:, :self.num_pesos] = np.random.uniform(
            low=-1.0, 
            high=1.0, 
            size=(self.tamanho_populacao, self.num_pesos)
        )
        
        # Inicializar aptidão com valores altos (não avaliados ainda)
        populacao[:, self.num_pesos] = np.inf
        
        return populacao
    
    def obter_individuo(self, indice: int) -> np.ndarray:
        """
        Obtém um indivíduo específico da população.
        
        Args:
            indice: Índice do indivíduo (0 a tamanho_populacao-1)
        
        Returns:
            individuo: Array com 181 elementos (180 pesos + 1 aptidão)
        """
        assert 0 <= indice < self.tamanho_populacao, f"Índice {indice} fora do intervalo"
        return self.populacao[indice].copy()
    
    def obter_pesos_individuo(self, indice: int) -> np.ndarray:
        """
        Obtém apenas os pesos de um indivíduo (sem a aptidão).
        
        Args:
            indice: Índice do indivíduo
        
        Returns:
            pesos: Array com 180 pesos
        """
        return self.populacao[indice, :self.num_pesos].copy()
    
    def obter_aptidao_individuo(self, indice: int) -> float:
        """
        Obtém a aptidão de um indivíduo específico.
        
        Args:
            indice: Índice do indivíduo
        
        Returns:
            aptidao: Valor da aptidão
        """
        return self.populacao[indice, self.num_pesos]
    
    def definir_aptidao_individuo(self, indice: int, aptidao: float):
        """
        Define a aptidão de um indivíduo específico.
        
        Args:
            indice: Índice do indivíduo
            aptidao: Valor da aptidão
        """
        self.populacao[indice, self.num_pesos] = aptidao
    
    def criar_rede_neural(self, indice: int) -> RedeNeural:
        """
        Cria uma rede neural com os pesos de um indivíduo específico.
        
        Args:
            indice: Índice do indivíduo
        
        Returns:
            rede: Instância da RedeNeural
        """
        pesos = self.obter_pesos_individuo(indice)
        return RedeNeural(pesos)
    
    def imprimir_estatisticas(self):
        """Imprime estatísticas da população."""
        print("📊 ESTATÍSTICAS DA POPULAÇÃO GENÉTICA")
        print("=" * 60)
        print(f"Tamanho da população: {self.tamanho_populacao}")
        print(f"Número de pesos por indivíduo: {self.num_pesos}")
        print(f"Colunas totais por indivíduo: {self.num_colunas}")
        
        # Estatísticas dos pesos
        todos_pesos = self.populacao[:, :self.num_pesos]
        print(f"\n🧬 ESTATÍSTICAS DOS PESOS:")
        print(f"   Mínimo: {todos_pesos.min():.4f}")
        print(f"   Máximo: {todos_pesos.max():.4f}")
        print(f"   Média: {todos_pesos.mean():.4f}")
        print(f"   Desvio padrão: {todos_pesos.std():.4f}")
        
        # Estatísticas das aptidões
        aptidoes = self.populacao[:, self.num_pesos]
        aptidoes_validas = aptidoes[aptidoes != np.inf]
        
        print(f"\n🎯 ESTATÍSTICAS DAS APTIDÕES:")
        if len(aptidoes_validas) > 0:
            print(f"   Melhor aptidão: {aptidoes_validas.min():.4f}")
            print(f"   Pior aptidão: {aptidoes_validas.max():.4f}")
            print(f"   Aptidão média: {aptidoes_validas.mean():.4f}")
            print(f"   Indivíduos avaliados: {len(aptidoes_validas)}/{self.tamanho_populacao}")
        else:
            print("   Nenhum indivíduo avaliado ainda")
        
        print("=" * 60)
    
    def imprimir_individuo(self, indice: int, precisao: int = 3):
        """
        Imprime detalhes de um indivíduo específico.
        
        Args:
            indice: Índice do indivíduo
            precisao: Número de casas decimais
        """
        assert 0 <= indice < self.tamanho_populacao, f"Índice {indice} fora do intervalo"
        
        pesos = self.obter_pesos_individuo(indice)
        aptidao = self.obter_aptidao_individuo(indice)
        
        print(f"\n🧠 INDIVÍDUO #{indice + 1}")
        print("=" * 40)
        print(f"Aptidão: {aptidao:.4f}")
        
        # Mostrar alguns pesos como exemplo
        print(f"Primeiros 10 pesos: {pesos[:10]}")
        print(f"Últimos 10 pesos: {pesos[-10:]}")
        
        # Estatísticas dos pesos deste indivíduo
        print(f"Range dos pesos: [{pesos.min():.{precisao}f}, {pesos.max():.{precisao}f}]")
        print(f"Média dos pesos: {pesos.mean():.{precisao}f}")
        print("=" * 40)


def exemplo_uso():
    """Demonstra o uso da população genética com redes neurais."""
    print("🧬 EXEMPLO DE POPULAÇÃO GENÉTICA + REDE NEURAL")
    print("=" * 70)
    
    # Criar população
    populacao = PopulacaoGenetica(tamanho_populacao=5)
    populacao.imprimir_estatisticas()
    
    # Mostrar alguns indivíduos
    print(f"\n📋 DETALHES DOS PRIMEIROS 3 INDIVÍDUOS:")
    for i in range(3):
        populacao.imprimir_individuo(i)
    
    # Testar uma rede neural com o primeiro indivíduo
    print(f"\n🎮 TESTE COM REDE NEURAL DO INDIVÍDUO #1:")
    rede = populacao.criar_rede_neural(0)
    
    # Tabuleiro de exemplo
    tabuleiro = np.array([1, 0, -1, 0, 1, 0, 0, -1, 0])
    
    print(f"Tabuleiro: {tabuleiro}")
    scores = rede.propagacao(tabuleiro)
    jogada = rede.escolher_jogada(tabuleiro)
    
    print(f"Scores: {scores}")
    print(f"Jogada escolhida: posição {jogada}")
    
    # Simular aptidão
    print(f"\n📈 SIMULANDO AVALIAÇÃO DE APTIDÃO:")
    for i in range(populacao.tamanho_populacao):
        # Simular aptidão aleatória (normalmente seria resultado de jogos)
        aptidao_simulada = np.random.uniform(0, 2)
        populacao.definir_aptidao_individuo(i, aptidao_simulada)
    
    # Mostrar estatísticas após avaliação
    print(f"\n📊 APÓS AVALIAÇÃO DE APTIDÃO:")
    populacao.imprimir_estatisticas()


if __name__ == "__main__":
    exemplo_uso() 