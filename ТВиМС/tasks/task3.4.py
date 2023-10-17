

def run(iters=17, p1=0.25, p2=0.75):
    probability = 0
    for i in range(1, iters+1):
        probability += p2 ** (i - 1) * p1

    print(f'Если сделать {iters} бросков, то буква А выпадет с вероятностью {probability}')
    


if __name__ == "__main__":
    run()