import numpy as np

bos_index = 26
eos_index = 27
pad_index = 28

sequence_length = 57
samples = np.load("/p/vast1/lbann/datasets/zinc/moses_zinc_train250K.npy", allow_pickle=True) 

train_samples = samples[:int(samples.size*0.8)]
#train_samples = samples[:256*4]
val_samples = samples[int(samples.size*0.8):int(samples.size*0.9)]
test_samples = samples[int(samples.size*0.9):]


# Train sample access functions
def get_train_sample(index):
    sample = train_samples[index]
    if len(sample) < sequence_length:
        sample = np.concatenate((sample, np.full(sequence_length-len(sample), pad_index)))
    else:
        sample = np.resize(sample, sequence_length)

    #sample_all = np.full(2*sequence_length, pad_index, dtype=int)
    #sample_all[0:len(sample)] = sample
    #sample_all[sequence_length:sequence_length+len(sample)] = sample

    return sample

# Validation sample access functions
def get_val_sample(index):
    sample = val_samples[index]
    if len(sample) < sequence_length:
        sample = np.concatenate((sample, np.full(sequence_length-len(sample), pad_index)))
    else:
        sample = np.resize(sample, sequence_length)

    #sample_all = np.full(2*sequence_length, pad_index, dtype=int)
    #sample_all[0:len(sample)] = sample
    #sample_all[sequence_length:sequence_length+len(sample)] = sample

    #return sample_all
    return sample

# Test sample access functions
def get_test_sample(index):
    sample = test_samples[index]
    if len(sample) < sequence_length:
        sample = np.concatenate((sample, np.full(sequence_length-len(sample), pad_index)))
    else:
        sample = np.resize(sample, sequence_length)

    sample_all = np.full(2*sequence_length, pad_index, dtype=int)
    sample_all[0:len(sample)] = sample
    sample_all[sequence_length:sequence_length+len(sample)] = sample

    #return sample_all
    return sample


def num_train_samples():
    return train_samples.shape[0]

def num_val_samples():
    return val_samples.shape[0]

def num_test_samples():
    return val_samples.shape[0]

def sample_dims():
    return (2*sequence_length+1,)

def vocab_size():
    return 30


