import random
import numpy as np
from typing import List, Tuple
import os
from .mlp import MLP
from .tic_tac_toe import JogoDaVelha
from .minimax import Minimax

class GeneticAlgorithm:
    def __init__(self):
        self.taxa_de_crossover = 0.0
        self.taxa_de_mutacao = 0.0
        self.numero_maximo_geracoes = 0
        self.tamanho_populacao = 0
        self.elitismo = True
        self.dificuldade = 1
        self.num_jogos_teste = 5  

    def run_ga(self) -> List[float]:
        geracao = 1
        
        populacao = self.Populacao(self.tamanho_populacao, True, self.dificuldade)
        populacao.ordena_populacao()
        
        print(f"Geração {geracao}:")
        print(f"Melhor: {populacao.get_individuo(0).get_aptidao()} ({populacao.get_individuo(0).get_aptidao()})")
        print(f"Média: {populacao.get_media_aptidao()}")
        print(f"Pior: {populacao.get_individuo(populacao.get_num_individuos() - 1).get_aptidao()} ({populacao.get_individuo(populacao.get_num_individuos() - 1).get_aptidao()})")
        print("-------------------------------------")

        while geracao < 20:  # Limited to 20 generations
            geracao += 1
            
            populacao = self.nova_geracao(populacao, geracao)
            
            if geracao == 10:  # Halfway point
                self.dificuldade = 2
                
            if geracao == 15:  # 75% point
                self.dificuldade = 4
                
            if geracao % 5 == 0:  # Print every 5 generations
                melhor = populacao.get_individuo(0).get_pesos()
                print(f"Geração {geracao}:")
                print(f"Aptidão do Melhor: {populacao.get_individuo(0).get_aptidao()}")
                print(f"{str(melhor)}")
            
            print(f"Geração {geracao}:")
            print(f"Melhor: {populacao.get_individuo(0).get_aptidao()} ({populacao.get_individuo(0).get_aptidao()})")
            print(f"Média: {populacao.get_media_aptidao()}")
            print(f"Pior: {populacao.get_individuo(populacao.get_num_individuos() - 1).get_aptidao()} ({populacao.get_individuo(populacao.get_num_individuos() - 1).get_aptidao()})")
            
            aptidoes = [populacao.get_individuo(i).get_aptidao() for i in range(populacao.get_tam_populacao())]
            print(" ".join(map(str, aptidoes)))
            print("-------------------------------------")
            
        print(f"Melhor indivíduo: {populacao.get_individuo(0).get_pesos()}")
        return populacao.get_individuo(0).get_pesos()

    def nova_geracao(self, populacao: 'Populacao', geracao: int) -> 'Populacao':
        nova_populacao = self.Populacao(populacao.get_tam_populacao(), False, self.dificuldade)
        
        if self.elitismo:
            nova_populacao.set_individuo(populacao.get_individuo(0))
            
        for i in range(nova_populacao.get_tam_populacao()):
            pais = [None, None]
            filho = None
            
            pais[0] = self.selecao_torneio(populacao, geracao)
            pais[1] = self.selecao_torneio(populacao, geracao)
            
            if random.random() <= self.taxa_de_crossover:
                filho = self.crossover(pais[0], pais[1])
            else:
                filho = pais[0]
                
            if i % 25 == 0:
                filho = self.mutacao(filho, geracao, self.numero_maximo_geracoes)
                
            nova_populacao.set_individuo(filho)
            
        nova_populacao.ordena_populacao()
        return nova_populacao

    def crossover(self, pai1: 'Individuo', pai2: 'Individuo') -> 'Individuo':
        filhos = self.Individuo(False, self.dificuldade)
        
        cromossomo1 = pai1.get_pesos()
        cromossomo2 = pai2.get_pesos()
        pesos_filho = np.zeros(180)
        
        for i in range(180):
            pesos_filho[i] = (cromossomo1[i] + cromossomo2[i]) / 2 + random.gauss(0, 0.1)
            
        filhos.set_pesos(pesos_filho)
        filhos.gera_aptidao(self.dificuldade)
        return filhos

    def selecao_torneio(self, populacao: 'Populacao', geracao: int) -> 'Individuo':
        tamanho_populacao = populacao.get_tam_populacao()
        
        if geracao < self.numero_maximo_geracoes * 0.25:
            limite = tamanho_populacao
        elif geracao < self.numero_maximo_geracoes * 0.75:
            limite = tamanho_populacao if random.random() < 0.3 else tamanho_populacao // 2
        else:
            limite = tamanho_populacao if random.random() < 0.3 else tamanho_populacao // 4
            
        indice1 = random.randint(0, limite - 1)
        indice2 = random.randint(0, limite - 1)
        while indice2 == indice1:
            indice2 = random.randint(0, limite - 1)
            
        individuo1 = populacao.get_individuo(indice1)
        individuo2 = populacao.get_individuo(indice2)
        
        return individuo1 if individuo1.get_aptidao() >= individuo2.get_aptidao() else individuo2

    def mutacao(self, individuo: 'Individuo', geracao_atual: int, numero_maximo_geracoes: int) -> 'Individuo':
        cromossomo = individuo.get_pesos()
        
        intensidade_mutacao = 1000 - (geracao_atual / numero_maximo_geracoes)
        
        for i in range(len(cromossomo)):
            if random.random() < self.taxa_de_mutacao:
                perturbacao = random.gauss(0, 1) * intensidade_mutacao * cromossomo[i]
                cromossomo[i] += perturbacao
                
        individuo.set_pesos(cromossomo)
        individuo.gera_aptidao(self.dificuldade)
        return individuo

    class Individuo:
        def __init__(self, inicializar: bool, dificuldade: int):
            self.pesos = np.zeros(180)
            self.aptidao = 0
            if inicializar:
                self.gera_pesos()
            self.gera_aptidao(dificuldade)
            
        def gera_pesos(self):
            self.pesos = np.random.uniform(-1, 1, 180)
            
        def gera_aptidao(self, dificuldade: int):
            
            rede = MLP(self.pesos)
            
            vitorias = 0
            empates = 0
            derrotas = 0
            
            for _ in range(GeneticAlgorithm().num_jogos_teste):
                resultado = self.jogar_partida(rede, dificuldade)
                if resultado == 1:
                    vitorias += 1
                elif resultado == 0:
                    empates += 1
                else:
                    derrotas += 1
            
            
            
            self.aptidao = (vitorias * 3 + empates) - derrotas
            
        def jogar_partida(self, rede: MLP, dificuldade: int) -> int:
            jogo = JogoDaVelha()
            turno = 1  
            
            while True:
                if turno == 1:  
                    jogada = rede.choose_move(jogo.tabuleiro)
                    if jogada is None:
                        return 0  
                    jogo.jogar(jogada)
                else:  
                    if dificuldade == 1:  
                        livres = [i for i in range(9) if jogo.tabuleiro[i] == 0]
                        if not livres:
                            return 0  
                        jogo.jogar(random.choice(livres))
                    else:  
                        minimax = Minimax("dificil" if dificuldade > 2 else "medio")
                        jogada = minimax.choose_move(jogo.tabuleiro)
                        if jogada is None:
                            return 0  
                        jogo.jogar(jogada)
                
                resultado = jogo.verificar_vencedor()
                if resultado is not None:
                    if resultado == 1:  
                        return 1
                    elif resultado == -1:  
                        return -1
                    else:  
                        return 0
                
                turno *= -1  
            
        def get_pesos(self) -> np.ndarray:
            return self.pesos
            
        def set_pesos(self, pesos: np.ndarray):
            self.pesos = pesos
            
        def get_aptidao(self) -> float:
            return self.aptidao

    class Populacao:
        def __init__(self, tamanho: int, inicializar: bool, dificuldade: int):
            self.individuos = []
            self.tamanho = tamanho
            if inicializar:
                for _ in range(tamanho):
                    self.individuos.append(GeneticAlgorithm.Individuo(True, dificuldade))
                    
        def get_individuo(self, index: int) -> 'Individuo':
            return self.individuos[index]
            
        def set_individuo(self, individuo: 'Individuo'):
            self.individuos.append(individuo)
            
        def ordena_populacao(self):
            self.individuos.sort(key=lambda x: x.get_aptidao(), reverse=True)
            
        def get_tam_populacao(self) -> int:
            return self.tamanho
            
        def get_num_individuos(self) -> int:
            return len(self.individuos)
            
        def get_media_aptidao(self) -> float:
            return sum(ind.get_aptidao() for ind in self.individuos) / len(self.individuos) 