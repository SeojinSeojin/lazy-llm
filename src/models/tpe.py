import numpy as np
import random
from src.utils.ezr import *
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from src.sampling.index import get_sampler

def TPE(args, sampler):
    i = DATA(csv(args.dataset))
    get_sampler(sampler)(i,args.label)

    # Extract X and Y from i.rows
    X = np.array([x[:len(i.cols.x)] for x in i.rows])
    Y = np.array([chebyshev(i, x) for x in i.rows])

    # Evaluation function
    def evaluate(params):
        idx = params['index']
        return {'loss': Y[idx], 'status': STATUS_OK}

    # Search space over dataset indices
    search_space = {
        'index': hp.choice('index', list(range(len(X))))
    }

    # Run Hyperopt
    trials = Trials()

    best = fmin(
        fn=evaluate,
        space=search_space,
        algo=tpe.suggest,
        max_evals=args.last,
        trials=trials,
    )

    # Best index and corresponding values
    best_idx = best['index']
    best_y = Y[best_idx]

    return best_y
