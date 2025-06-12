from core.genetic_algorithm import GA

def train():
    ag = GA(tamanho_pop=30)
    for gen in range(100):
        ag.evaluate(modo=choose_mode(gen))
        print(
            f"Geração {gen} - melhor aptidão: {max(ind.aptidao for ind in ag.populacao)}"
        )
        ag.evolve()


def choose_mode(g):
    if g < 50:
        return "medio"
    else:
        return "dificil"


if __name__ == "__main__":
    train()
