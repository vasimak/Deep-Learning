from multiprocessing import Pool
from train_utils import *

if __name__ == '__main__':
    with Pool(processes=4) as pool:
        results = [pool.apply_async(train_step_wrapper, args=(x, y)) for x, y in zip(X_train, y_train)]
        [r.get() for r in results]

