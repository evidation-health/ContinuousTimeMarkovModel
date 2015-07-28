import pymc3 as pm

K = 10
D = 250

model = pm.Model()
with model:
    Q_raw = DiscreteObsMJP_unif_prior('Q_raw', n=5, shape=(5,4))
    
    S = HMM_Blank('S')
    C = DiscreteObsMJP('C', Q_raw=Q_raw, S=S, step_sizes=step_sizes)

    B = Beta('B', )
    X = Comorbidities('X', S=S, B=B, shape=(K, T_n))

    Z = Beta('Z')
    L = Beta('L')
    O = Observations('O', X=X, Z=Z, L=L, shape=(D, T_n))

with model:
    step1 = Metropolis(vars=[Q_raw])
    step2 = sampleS(vars=[S])

    step3 = Metropolis(vars=[B])
    step4 = sampleX(vars=[X])