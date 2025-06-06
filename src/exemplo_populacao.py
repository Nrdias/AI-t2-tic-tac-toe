#!/usr/bin/env python3
"""
Exemplo específico da matriz de população do algoritmo genético
conforme mostrado na imagem.

Estrutura da matriz:
- Cada linha = 1 indivíduo 
- Colunas 0...179 = 180 pesos da rede neural
- Coluna 180 = função aptidão
- Valores aleatórios entre -1 e 1 para os pesos

Author: Baseado na imagem fornecida
Date: 2025
"""

import numpy as np
from rede_neural_genetica import PopulacaoGenetica, RedeNeural


def mostrar_estrutura_matriz():
    """Mostra a estrutura da matriz exatamente como na imagem."""
    print("🧬 ESTRUTURA DA MATRIZ DA POPULAÇÃO")
    print("=" * 70)
    
    # Criar população pequena para demonstração
    populacao = PopulacaoGenetica(tamanho_populacao=3)
    matriz = populacao.populacao
    
    print("Dimensões da matriz:", matriz.shape)
    print("Linhas (indivíduos):", matriz.shape[0])
    print("Colunas (pesos + aptidão):", matriz.shape[1])
    
    print("\n📋 ESTRUTURA DAS COLUNAS:")
    print("  Colunas 0...179  → 180 pesos da rede neural")
    print("  Coluna 180       → função aptidão")
    
    print(f"\n🔢 PRIMEIROS VALORES DE CADA INDIVÍDUO:")
    for i in range(populacao.tamanho_populacao):
        print(f"\n  Indivíduo {i}:")
        # Mostrar primeiros pesos (como na imagem: -0.1, -0.008, -0.5, 0.0625, ...)  
        primeiros_pesos = matriz[i, :10]
        aptidao = matriz[i, 180]
        
        print(f"    Primeiros 10 pesos: [{', '.join([f'{p:.3f}' for p in primeiros_pesos])}]")
        print(f"    Aptidão (coluna 180): {aptidao}")
    
    return matriz


def demonstrar_exemplo_imagem():
    """Cria um exemplo que mostra valores similares aos da imagem."""
    print("\n" + "=" * 70)
    print("🖼️  EXEMPLO SIMULANDO OS VALORES DA IMAGEM")
    print("=" * 70)
    
    # Criar uma população pequena
    populacao = PopulacaoGenetica(tamanho_populacao=2)
    
    # Simular alguns valores específicos como na imagem
    # Indivíduo 1: começando com -0.1, -0.008, -0.5, 0.0625, ...
    exemplo_pesos_1 = np.array([-0.1, -0.008, -0.5, 0.0625] + 
                              [np.random.uniform(-1, 1) for _ in range(176)])
    populacao.populacao[0, :180] = exemplo_pesos_1
    populacao.definir_aptidao_individuo(0, 0.85)  # Aptidão exemplo
    
    # Mostrar formatação similar à imagem
    print("população")
    print("┌" + "─" * 50 + "┬" + "─" * 5 + "┐")
    print("│" + " " * 18 + "0 … 179" + " " * 18 + "│ 180 │")
    print("├" + "─" * 50 + "┼" + "─" * 5 + "┤")
    
    # Primeira linha (como na imagem)
    pesos_str = " | ".join([f"{p:.3f}" for p in exemplo_pesos_1[:4]]) + " | …"
    aptidao_str = f"{populacao.obter_aptidao_individuo(0):.2f}"
    print(f"│ {pesos_str:<48} │{aptidao_str:>4} │")
    
    # Segunda linha  
    individuo_2 = populacao.obter_pesos_individuo(1)
    pesos_str_2 = " | ".join([f"{p:.3f}" for p in individuo_2[:4]]) + " | …"
    populacao.definir_aptidao_individuo(1, 1.20)
    aptidao_str_2 = f"{populacao.obter_aptidao_individuo(1):.2f}"
    print(f"│ {pesos_str_2:<48} │{aptidao_str_2:>4} │")
    
    print("└" + "─" * 50 + "┴" + "─" * 5 + "┘")
    
    print(f"\n📊 DETALHES TÉCNICOS:")
    print(f"  • Cada linha = 1 indivíduo (rede neural)")
    print(f"  • 180 pesos por indivíduo organizados como:")
    print(f"    - W1 (9×9): posições 0-80   (81 pesos)")
    print(f"    - b1 (9):   posições 81-89  (9 bias)")
    print(f"    - W2 (9×9): posições 90-170 (81 pesos)")
    print(f"    - b2 (9):   posições 171-179 (9 bias)")
    print(f"  • Coluna 180: aptidão (quanto menor, melhor)")
    
    return populacao


def testar_propagacao_com_individuo():
    """Testa a propagação usando um indivíduo da população."""
    print("\n" + "=" * 70)
    print("🧠 TESTE DE PROPAGAÇÃO COM INDIVÍDUO DA POPULAÇÃO")
    print("=" * 70)
    
    # Usar a população criada anteriormente
    populacao = PopulacaoGenetica(tamanho_populacao=1)
    
    # Criar rede neural com os pesos do primeiro indivíduo
    rede = populacao.criar_rede_neural(0)
    
    print(f"✅ Rede neural criada com pesos do indivíduo #1")
    print(f"   Total de parâmetros: 180")
    
    # Exemplo de entrada (tabuleiro do jogo da velha)
    tabuleiro = np.array([1, 0, -1, 0, 0, 0, 1, -1, 0])
    print(f"\n🎮 Testando com tabuleiro:")
    print(f"   Entrada: {tabuleiro}")
    print(f"   Formato visual:")
    print(f"     {tabuleiro[0]:2} | {tabuleiro[1]:2} | {tabuleiro[2]:2}")
    print(f"     -----------")  
    print(f"     {tabuleiro[3]:2} | {tabuleiro[4]:2} | {tabuleiro[5]:2}")
    print(f"     -----------")
    print(f"     {tabuleiro[6]:2} | {tabuleiro[7]:2} | {tabuleiro[8]:2}")
    
    # Executar propagação
    saidas = rede.propagacao(tabuleiro)
    jogada = rede.escolher_jogada(tabuleiro)
    
    print(f"\n⚡ Resultado da propagação:")
    print(f"   Saídas: {saidas}")
    print(f"   Jogada escolhida: posição {jogada}")
    
    # Mostrar detalhes da arquitetura
    print(f"\n🏗️  ARQUITETURA DA REDE:")
    print(f"   Entrada: 9 neurônios (posições do tabuleiro)")
    print(f"   Camada oculta: 9 neurônios (ativação tanh)")
    print(f"   Camada saída: 9 neurônios (ativação linear)")
    print(f"   Total conexões: 9×9 + 9×9 = 162 pesos + 18 bias = 180 parâmetros")


def main():
    """Função principal demonstrando todos os conceitos."""
    print("🎯 DEMONSTRAÇÃO COMPLETA DA REDE NEURAL + ALGORITMO GENÉTICO")
    print("=" * 70)
    
    # 1. Mostrar estrutura da matriz
    matriz = mostrar_estrutura_matriz()
    
    # 2. Exemplo similar à imagem
    populacao = demonstrar_exemplo_imagem()
    
    # 3. Teste de propagação
    testar_propagacao_com_individuo()
    
    print(f"\n✅ RESUMO:")
    print(f"  ✓ Matriz da população criada: {matriz.shape}")
    print(f"  ✓ Pesos aleatórios entre -1 e 1: ✓")
    print(f"  ✓ 180 pesos + 1 aptidão por indivíduo: ✓")
    print(f"  ✓ Rede neural MLP 9→9→9 implementada: ✓")
    print(f"  ✓ Propagação forward funcionando: ✓")
    
    print(f"\n🚀 A implementação está pronta para ser usada com algoritmo genético!")


if __name__ == "__main__":
    main() 