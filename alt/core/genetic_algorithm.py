import random
import numpy as np
from .individual import Individual
from .tic_tac_toe import JogoDaVelha
from .minimax import Minimax


class GA:
    def __init__(self, tamanho_pop=50):
        self.populacao = [Individual() for _ in range(tamanho_pop)]
        self.minimax = Minimax(mode="medio")  # Começa com modo médio

    def evaluate(self, modo="hard"):
        """
        Avalia cada indivíduo da população fazendo ele jogar contra o Minimax
        
        Args:
            modo (str): "medio" ou "dificil" - modo do Minimax
        """
        # Atualiza o modo do Minimax
        self.minimax.mode = modo
        
        for ind in self.populacao:
            rede = ind.get_network()
            pontos = 0
            for _ in range(5):  # Cada rede joga 5 partidas
                jogo = JogoDaVelha()
                fim = None
                while fim is None:
                    # Vez da rede neural
                    jogada = rede.choose_move(jogo.tabuleiro)
                    if jogada is None or not jogo.jogar(jogada):
                        pontos -= 1  # penaliza jogada inválida
                        break
                    
                    fim = jogo.verificar_vencedor()
                    if fim is not None:
                        break
                    
                    # Vez do Minimax
                    jogada_op = self.minimax.choose_move(jogo.tabuleiro)
                    jogo.jogar(jogada_op)
                    fim = jogo.verificar_vencedor()
                
                # Pontuação baseada no resultado
                if fim == 1:  # Vitória da rede
                    pontos += 1
                elif fim == 0:  # Empate
                    pontos += 0.5
                # Derrota não ganha pontos
            
            ind.aptidao = pontos

    def evolve(self):
        """
        Evolui a população usando:
        - Elitismo: mantém os 10 melhores
        - Seleção por torneio: escolhe entre os 20 melhores
        - Cruzamento: combina pesos dos pais
        - Mutação: adiciona ruído gaussiano
        """
        # Ordena por aptidão
        self.populacao.sort(key=lambda x: x.aptidao, reverse=True)
        
        # Elitismo: mantém os 10 melhores
        nova_geracao = self.populacao[:10]
        
        # Completa a população
        while len(nova_geracao) < len(self.populacao):
            # Seleção por torneio entre os 20 melhores
            pai, mae = random.sample(self.populacao[:20], 2)
            
            # Cruzamento: combina os pesos dos pais
            ponto = random.randint(1, len(pai.pesos) - 1)
            filho_pesos = np.concatenate((pai.pesos[:ponto], mae.pesos[ponto:]))
            
            # Mutação: adiciona ruído gaussiano
            filho_pesos += np.random.normal(0, 0.1, size=filho_pesos.shape)
            
            # Adiciona o novo indivíduo
            nova_geracao.append(Individual(filho_pesos))
        
        self.populacao = nova_geracao
