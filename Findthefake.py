import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
test_path = 'C:/Users/Public/Downloads/train.csv'
train_df = pd.read_csv("C:/Users/Public/Downloads/train.csv")
test_df = pd.read_csv("C:/Users/Public/Downloads/test.csv")
test_df.drop(['ID_code'], axis=1, inplace=True)
test_df = test_df.values
unique_samples = []
unique_count = np.zeros_like(test_df)

for feature in range(test_df.shape[1]):
    _, index_, count_ = np.unique(test_df[:, feature], return_index=True, return_counts=True)
    unique_count[index_[count_ == 1], feature] += 1

real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]
print(len(real_samples_indexes))
print(len(synthetic_samples_indexes))

# We create a new generator
test_df_real = test_df[real_samples_indexes].copy()
generator_for_each_synthetic_sample = []
for cur_sample_index in synthetic_samples_indexes[:20000]:
    cur_synthetic_sample = test_df[cur_sample_index]
    potential_generators = test_df_real == cur_synthetic_sample
    features_mask = np.sum(potential_generators, axis=0) == 1
    verified_generators_mask = np.any(potential_generators[:, features_mask], axis=1)
    verified_generators_for_sample = real_samples_indexes[np.argwhere(verified_generators_mask)[:, 0]]
    generator_for_each_synthetic_sample.append(set(verified_generators_for_sample))

public_LB = generator_for_each_synthetic_sample[0]
for x in generator_for_each_synthetic_sample:
    if public_LB.intersection(x):
        public_LB = public_LB.union(x)
private_LB = generator_for_each_synthetic_sample[1]
for x in generator_for_each_synthetic_sample:
    if private_LB.intersection(x):
        private_LB = private_LB.union(x)
print(len(public_LB))
print(len(private_LB))
