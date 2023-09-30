from helper import neutralize, normalize

class Alpha:
    def __init__(self, alpha_states) -> None:
        self.alpha_matrix = normalize(neutralize(alpha_states))

    

