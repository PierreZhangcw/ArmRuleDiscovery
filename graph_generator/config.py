


class Config:
    def __init__(self):
        
        # 1. parameter for elements
        self.number_entity = 100
        self.number_relation = 100
        self.alpha_head = 0.7
        self.alpha_tail = 0.5
        
        # 2. parameter for generating arms
        self.beta = 0.01
        
        # 3. parameter for generating rules
        self.delta_arm = 0.5
        self.delta_rel = 0.4
        self.ratio_rr = 0.1   # rules type rr/relations
        self.ratio_ra = 0.1 # rules type ra/relations
        self.ratio_ar = 0.1 # rules type ar/relations
        self.d = 6    # rules type aa for each arm
        self.epsilon_random = 0.1
        self.epsilon_miss = 0.01
