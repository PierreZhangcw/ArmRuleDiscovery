import multiprocessing
import numpy as np
from tqdm import tqdm
import sys
import random
import copy
import csv
import pickle
from config import *


class graph:
    def __init__(self,config):
        self.config = config
        self.generate_element()
        self.generate_arm()
        self.generate_rules()
        
    def generate_element(self):
        # 1. entities and relations
        ## results of this part: H,T,R 
        # parameters for this part
        # entity set
        V = list(range(self.config.number_entity))
        self.H = set(V[:int(self.config.number_entity*self.config.alpha_head)])
        self.T = set(V[self.config.number_entity-int(self.config.number_entity*self.config.alpha_tail):])
        # relation set
        self.R = set(list(range(self.config.number_relation)))

    def generate_arm(self):
        # 2. arm 
        ## results of this part: H, T, A
        # parameters for this part
        rel_tail = {}
        used_tail = set()
        self.A = set()
        for r in self.R:
            for t in self.T:
                if np.random.rand()<self.config.beta:
                    if r not in rel_tail:
                        rel_tail[r] = set()
                    rel_tail[r].add(t)
                    self.A.add((r,t))
                    used_tail.add(t)

        self.H = self.H|(self.T-used_tail)
        self.T = used_tail
        
    def generate_rules(self):
        self.generate_effect()
        self.generate_rule_rr()
        self.generate_rule_ra()
        self.generate_rule_ar()
        self.generate_rule_aa()
        self.combine_rule()
        G_max = self.generate_G_max()
        G_min = self.generate_G_min(G_max)
        
        self.G_max, self.G_min = set(),set()
        i = 0 
        for h in self.H:
            for a in G_max[i]:
                self.G_max.add((h,a))
            for a in G_min[i]:
                self.G_min.add((h,a))
            i+=1
            
    def generate_effect(self):
        # effect_arms
        self.effect_pos_arm = set()
        self.effect_neg_arm = set()
        for a in self.A:
            if np.random.rand()<self.config.delta_arm:
                self.effect_pos_arm.add(a)
            if np.random.rand()<self.config.delta_arm:
                self.effect_neg_arm.add(a)
        self.effect_intersec_arm = self.effect_pos_arm&self.effect_neg_arm
        self.effect_arm = self.effect_pos_arm|self.effect_neg_arm
        # effect relations
        self.effect_pos_rel = set()
        self.effect_neg_rel = set()
        for r in self.R:
            if np.random.rand()<self.config.delta_rel:
                self.effect_pos_rel.add(r)
            if np.random.rand()<self.config.delta_rel:
                self.effect_neg_rel.add(r)
        self.effect_intersec_rel = self.effect_pos_rel&self.effect_neg_rel
        self.effect_rel = self.effect_pos_rel|self.effect_neg_rel
        
    def transitive_closure(self,closure):
        closure_dic = {}
        for rule in closure:
            if rule[0] not in closure_dic:
                closure_dic[rule[0]] = set()
            closure_dic[rule[0]].add(rule[1])
        new = True
        while new:
            new = False
            for a1,a2 in copy.deepcopy(closure):
                if a2 in closure_dic:
                    for a3 in closure_dic[a2]:
                        if a3 not in closure_dic[a1]:
                            new = True
                            closure_dic[a1].add(a3)
                            closure.add((a1,a3))
        return closure
        
    def generate_rule_rr(self):
        # 3.1 rules of type-RR
        k_rr = int(len(self.R)*self.config.ratio_rr)
        cause_rr = random.sample(self.R,k_rr)
        effect_rr = random.sample(self.effect_rel,k_rr)
        self.rel_rel_pos, self.rel_rel_neg = set(), set()
        for i in range(k_rr):
            if cause_rr[i]==effect_rr:
                continue
            if effect_rr[i] in self.effect_intersec_rel:
                if np.random.rand()<0.5:
                    self.rel_rel_pos.add((cause_rr[i],effect_rr[i]))
                else:
                    self.rel_rel_neg.add((cause_rr[i],effect_rr[i]))
            elif effect_rr[i] in self.effect_pos_rel:
                self.rel_rel_pos.add((cause_rr[i],effect_rr[i]))
            else:
                self.rel_rel_neg.add((cause_rr[i],effect_rr[i]))

        # delete conflict
        # (1) first time for deleting conflict rules
        delete_pos_rr,delete_neg_rr = set(), set()
        for rule in self.rel_rel_pos:
            if (rule[-1],rule[0]) in self.rel_rel_neg:
                if np.random.rand()<0.5:
                    delete_pos_rr.add(rule)
                else:
                    delete_neg_rr.add((rule[-1],rule[0]))
        self.rel_rel_pos = self.rel_rel_pos - delete_pos_rr
        self.rel_rel_neg = self.rel_rel_neg - delete_neg_rr
        # (2) secondly, closure for negative rules
        for rule in copy.deepcopy(self.rel_rel_neg):
            self.rel_rel_neg.add((rule[-1],rule[0]))
        # (3) closure for positive rules
        self.rel_rel_pos = self.transitive_closure(self.rel_rel_pos)

        # (4) delete conflict negative rules
        delete_neg_rr = set()
        for rule in self.rel_rel_neg&self.rel_rel_pos:
            delete_neg_rr.add(rule)
            delete_neg_rr.add((rule[1],rule[0]))
        self.rel_rel_neg = self.rel_rel_neg - delete_neg_rr
        
    def generate_rule_ra(self):
        # 3.2 rules of type-RA and type-AR
        k_ra = int(len(self.R)*self.config.ratio_ra)
        cause_ra = random.sample(self.R,k_ra)
        effect_ra = random.sample(self.effect_arm,k_ra)
        self.rel_arm_pos, self.rel_arm_neg = set(), set()
        for i in range(k_ra):
            r = cause_ra[i]
            arm = effect_ra[i]
            if r==arm[0] or (r,arm[0]) in self.rel_rel_pos or (r,arm[0]) in self.rel_rel_neg:
                continue
            if arm in self.effect_intersec_arm:
                if np.random.rand()<0.5:
                    self.rel_arm_pos.add((r,arm))
                else:
                    self.rel_arm_neg.add((r,arm))
            elif arm in self.effect_pos_arm:
                self.rel_arm_pos.add((r,arm))
            else:
                self.rel_arm_neg.add((r,arm))

    def generate_rule_ar(self):
        # 3.3 rules of type-AR
        k_ar = int(len(self.R)*self.config.ratio_ar)
        cause_ar = random.sample(self.A,k_ar)
        effect_ar = random.sample(self.effect_rel,k_ar)
        self.arm_rel_pos, self.arm_rel_neg = set(), set()
        for i in range(k_ar):
            r = effect_ar[i]
            arm = cause_ar[i]
            if r==arm[0] or (arm[0],r) in self.rel_rel_pos or (arm[0],r) in self.rel_rel_neg:
                continue
            if r in self.effect_intersec_rel:
                if np.random.rand()<0.5:
                    self.arm_rel_pos.add((arm,r))
                else:
                    self.arm_rel_neg.add((arm,r))
            elif arm in self.effect_pos_arm:
                self.arm_rel_pos.add((arm,r))
            else:
                self.arm_rel_neg.add((arm,r))
                
        # delete conflict
        # (1) first time for deleting conflict rules
        delete_pos_ra, delete_neg_ra = set(), set()
        delete_pos_ar, delete_neg_ar = set(), set()
        for r,a in self.rel_arm_neg:
            if (a,r) in self.arm_rel_pos:
                if np.random.rand()<0.5:
                    delete_pos_ar.add((a,r))
                else:
                    delete_neg_ra.add((r,a))

        for a,r in self.arm_rel_neg:
            if (r,a) in self.rel_arm_pos:
                if np.random.rand()<0.5:
                    delete_pos_ra.add((r,a))
                else:
                    delete_neg_ar.add((a,r))

        self.rel_arm_pos = self.rel_arm_pos - delete_pos_ra
        self.rel_arm_neg = self.rel_arm_neg - delete_neg_ra

        self.arm_rel_pos = self.arm_rel_pos - delete_pos_ar
        self.arm_rel_neg = self.arm_rel_neg - delete_neg_ar

    def generate_rule_aa(self):
        # 3.4 rules of type-AA
        self.arm_arm_pos,self.arm_arm_neg = set(), set()
        for a1 in self.A:
            effect_arm_aa = random.sample(self.effect_arm,self.config.d)
            for a2 in effect_arm_aa:
                if (a1[0],a2[0]) in self.rel_rel_pos or (a1[0],a2[0]) in self.rel_rel_neg:
                    continue
                if (a1[0],a2) in self.rel_arm_pos or (a1[0],a2) in self.rel_arm_neg:
                    continue
                if (a1,a2[0]) in self.arm_rel_pos or (a1,a2[0]) in self.arm_rel_neg:
                    continue
                if a2 in self.effect_intersec_arm:
                    if np.random.rand()<0.5:
                        self.arm_arm_pos.add((a1,a2))
                    else:
                        self.arm_arm_neg.add((a1,a2))
                elif a2 in self.effect_pos_arm:
                    self.arm_arm_pos.add((a1,a2))
                else:
                    self.arm_arm_neg.add((a1,a2))

    def combine_rule(self):
        # (1) closure of positive rules
        self.rule_pos = self.rel_rel_pos|self.rel_arm_pos|self.arm_rel_pos|self.arm_arm_pos
        self.rule_pos = self.transitive_closure(self.rule_pos)

        # (2) closure of negative rules
        self.rule_neg = self.rel_rel_neg|self.rel_arm_neg|self.arm_rel_neg|self.arm_arm_neg
        for rule in copy.deepcopy(self.rule_neg):
            self.rule_neg.add((rule[-1],rule[0]))

        # (3) delete conflict negative rules
        delete_neg = set()
        for rule in self.rule_pos:
            if rule in self.rule_neg:
                delete_neg.add(rule)
                delete_neg.add((rule[-1],rule[0]))
        self.rule_neg = self.rule_neg - delete_neg

    # 4.Gmax
    def generate_max_one(self,h):
        # A should be a list
        A = list(self.A)
        while True:
            mat = np.zeros(shape=(len(A)))
            for i in range(len(A)):
                a = A[i]
                for j in range(i + 1, len(A)):
                    b = A[j]
                    if (a,b) in self.rule_neg or (a[0],b) in self.rule_neg or (a,b[0]) in self.rule_neg or (a[0],b[0]) in self.rule_neg:
                        mat[i] += 1
                        mat[j] += 1
            if np.count_nonzero(mat) == 0:
                return set(A)
            alis = np.argsort(mat)[-1:-6:-1]
            i = random.choice(alis)
            A.remove(A[i])
    def generate_G_max(self):
        G_max = []
        pool = multiprocessing.Pool()
        results = pool.map(self.generate_max_one,self.H)
        pool.close()
        pool.join()
        for res in results:
            G_max.append(res)
        #print(G_max)
        return G_max

    # 5. Gmin
    def generate_min_one(self,A):
        A = list(A)
        used_arm = set()
        for i in range(len(A)):
            a = A[i]
            for j in range(i + 1, len(A)):
                b = A[j]
                if (a,b) in self.rule_pos or (a[0],b) in self.rule_pos or (a,b[0]) in self.rule_pos or (a[0],b[0]) in self.rule_pos:
                    used_arm.add(a)
                    used_arm.add(b)
        return used_arm
    
    def generate_G_min(self,G_max):
        G_min = []
        pool = multiprocessing.Pool()
        results = pool.map(self.generate_min_one,G_max)
        pool.close()
        pool.join()
        for res in results:
            G_min.append(res)
        return G_min

    def G_fact(self,epsilon_random=None):
        if not epsilon_random:
            epsilon_random = self.config.epsilon_random
        G_complement = self.G_max-self.G_min
        random_triples = random.sample(G_complement,int(len(G_complement)*epsilon_random))
        G.facts = set(random_triples)|self.G_min
        return G.facts
    
    def G_obs(self,facts,epsilon_miss=None):
        if not epsilon_miss:
            epsilon_miss=self.config.epsilon_miss
        return facts-set(random.sample(facts,int(len(facts)*epsilon_miss)))
    def save_triples(self,data,filename = "facts",mode = 'csv'):
        if mode=="txt":
            with open("data/"+filename+".txt","a+") as f:
                for triple in data:
                    line = '\t'.join([str(triple[0]),str(triple[1][0]),str(triple[1][1])])+"\n"
                    f.write(line)
        elif mode == "csv":
            with open("data/"+filename+".csv","a+") as f:
                writer = csv.writer(f)
                writer.writerow(['head','relation','tail','arm'])
                for triple in data:
                    head,arm = triple
                    relation,tail = arm
                    writer.writerow([head,relation,tail,arm])
    def save_rules():
        with open("data/true_pos_rule.pkl","wb") as f:
            pickle.dump(self.rule_pos,f)
        with open("data/true_neg_rule.pkl","wb") as f:
            pickle.dump(self.rule_neg,f)
        
                                      
if __name__=='__main__':
    config = Config()
    G = graph(config)
    #print(G.G_min)
    print(len(G.G_max),len(G.G_min))
    facts = G.G_fact()
    observe = G.G_obs(facts)
    G.save_triples(observe,"triples")
    print(len(facts),len(observe))
    with open("data/my_graph.pkl",'wb') as f:
        pickle.dump(G,f)
