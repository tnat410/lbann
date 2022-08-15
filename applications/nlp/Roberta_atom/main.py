from types import SimpleNamespace
import argparse
import datetime
import os
import sys
import json
import numpy as np

import lbann
import lbann.contrib.args
import lbann.contrib.launcher

from lbann.models import RoBERTaMLM


# Local imports
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
import utils.paths



import dataset
# Dataset properties
vocab_size = dataset.vocab_size()
sequence_length = dataset.sequence_length
pad_index = dataset.pad_index
ignore_index = dataset.ignore_index

# ----------------------------------------------
# Options
# ----------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--epochs",
    default=51,
    type=int,
    help="number of epochs to train",
)
parser.add_argument(
    "--mini-batch-size",
    default=256,
    type=int,
    help="size of minibatches for training",
)
parser.add_argument(
    "--job-name",
    action="store",
    default="RoBERTa_MLM",
    type=str,
    help="scheduler job name",
    metavar="NAME",
)
parser.add_argument(
    "--work-dir",
    action="store",
    default=None,
    type=str,
    help="working directory",
    metavar="DIR",
)
parser.add_argument("--batch-job", action="store_true", help="submit as batch job")
parser.add_argument(
    "--checkpoint", action="store_true", help="checkpoint trainer after every epoch"
)
lbann.contrib.args.add_scheduler_arguments(parser)
lbann_params = parser.parse_args()



# ----------------------------------------------
# Work directory
# ----------------------------------------------

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
work_dir = os.path.join(
    utils.paths.root_dir(),
    'Roberta_atom/exps',
    f'{timestamp}_{lbann_params.job_name}',
)
os.makedirs(work_dir, exist_ok=True)



# ----------------------------------------------
# Data Reader
# ----------------------------------------------
def make_data_reader():
    reader = lbann.reader_pb2.DataReader()

    # Train data reader
    _reader = reader.reader.add()
    _reader.name = "python"
    _reader.role = "train"
    _reader.shuffle = True
    _reader.percent_of_data_to_use = 1.0
    _reader.python.module = "dataset"
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = "get_train_sample"
    _reader.python.num_samples_function = "num_train_samples"
    _reader.python.sample_dims_function = "sample_dims"

    # Validation data reader
    _reader = reader.reader.add()
    _reader.name = "python"
    _reader.role = "validate"
    _reader.shuffle = False
    _reader.percent_of_data_to_use = 1.0
    _reader.python.module = "dataset"
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = "get_val_sample"
    _reader.python.num_samples_function = "num_val_samples"
    _reader.python.sample_dims_function = "sample_dims"

    # Test data reader
    _reader = reader.reader.add()
    _reader.name = "python"
    _reader.role = "test"
    _reader.shuffle = False
    _reader.percent_of_data_to_use = 1.0
    _reader.python.module = "dataset"
    _reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
    _reader.python.sample_function = "get_test_sample"
    _reader.python.num_samples_function = "num_test_samples"
    _reader.python.sample_dims_function = "sample_dims"

    return reader



# ----------------------------------------------
# Build and Run Model
# ----------------------------------------------
with open("./config.json") as f:
    config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
config.input_shape = (1,sequence_length)

config.load_weights = os.path.abspath('./pretrained_weights')


# Construct the model

# Input is 3 sequences of smile string: original string, masked string, label string - every token is -100 (ignore) except the masked token. 
input_ = lbann.Input(data_field='samples')


input_strings = lbann.Identity(lbann.Slice(
	input_,
	axis=0,
	slice_points=[0,sequence_length],
	name='input_strings'
))

input_masked = lbann.Identity(lbann.Slice(
	input_,
	axis=0,
	slice_points=[sequence_length,2*sequence_length],
	name='input_masked'
))

input_label = lbann.Identity(lbann.Slice(
	input_,
	axis=0,
	slice_points=[2*sequence_length,3*sequence_length],
	name='input_label'
))


############
# Masking
############


value = 1 - 0.15
mask_percent_weights = lbann.Weights(initializer=lbann.ValueInitializer(values=value), name='mask_percent_weights', optimizer=lbann.NoOptimizer())
mask_percent = lbann.WeightsLayer(dims=1, weights=mask_percent_weights, name='mask_percent', device='CPU')
#mask_percent = lbann.Reduction(mask_percent, mode='sum')

rand = lbann.Uniform(min = 0, max = 1, neuron_dims = sequence_length,training_only=True)

rand_value = lbann.Identity(rand,name='rand_value')

half = lbann.Constant(value=0.5, num_neurons=sequence_length)
train_flag = lbann.NotEqual(rand_value, half)
test_flag = lbann.Equal(rand_value, half)

bos = lbann.Constant(value=bos_index, num_neurons=sequence_length)
not_bos_index = lbann.NotEqual(input_strings, bos)

eos = lbann.Constant(value=eos_index, num_neurons=sequence_length)
not_eos_index = lbann.NotEqual(input_strings, eos)

pad = lbann.Constant(value=pad_index, num_neurons=sequence_length)
not_pad_index = lbann.NotEqual(input_strings, pad)

mask = lbann.Tessellate(mask_percent,dims=sequence_length)
less_than_mask_percent = lbann.GreaterEqual(rand,mask)

# index that is not bos, eos, pad, and mask_prob < 15
special_index = lbann.Multiply(not_bos_index,not_eos_index)
special_index = lbann.Multiply(special_index,not_pad_index,name="special_index")
replace_index = lbann.Multiply(less_than_mask_percent,special_index,name="replace_index")

zero = lbann.Constant(value=0, num_neurons=sequence_length)

#Mask
no_mask_index = lbann.Equal(replace_index, zero)
no_mask_input = lbann.Multiply(input_strings,no_mask_index)

mask = lbann.Constant(value = mask_index, num_neurons=sequence_length)
mask_idx = lbann.Multiply(mask,replace_index,name="mask_index")
input_masked_train = lbann.Sum(no_mask_input,mask_idx)

#Label
no_ignore_index = lbann.Equal(replace_index, zero)
no_ignore_input = lbann.Multiply(input_strings,replace_index,name="no_ignore_input")

ignore = lbann.Constant(value = ignore_index, num_neurons = sequence_length)
ignore_idx = lbann.Multiply(ignore,no_ignore_index,name="ignore_index")
input_label_train = lbann.Sum(no_ignore_input,ignore_idx)

#Train/test data
train_input = lbann.Multiply(input_masked_train,train_flag)
test_input = lbann.Multiply(input_aug,test_flag)

input_masked = lbann.Sum(train_input,test_input)

train_input_label = lbann.Multiply(input_label_train,train_flag)
test_input_label = lbann.Multiply(input_label_ori,test_flag)

input_label = lbann.Sum(train_input_label,test_input_label)

input_masked_out = lbann.Identity(input_masked,name='input_masked')
input_label_out = lbann.Identity(input_label,name='input_label')

########
# Model
########

robertamlm = RoBERTaMLM(config,load_weights=config.load_weights)
output = robertamlm(input_masked)

preds_output = lbann.Identity(output,name='pred')

preds = lbann.ChannelwiseSoftmax(output, name='pred_sm')
preds = lbann.Slice(preds, axis=1, slice_points=range(sequence_length+1),name='slice_pred')
preds = [lbann.Identity(preds) for _ in range(sequence_length)]


########
# Loss
########

# Count number of non-pad tokens
label_tokens = lbann.Identity(input_label)
pads = lbann.Constant(value=ignore_index, num_neurons=sequence_length,name='pads')

is_not_pad = lbann.NotEqual(label_tokens, pads,name='is_not_pad')
num_not_pad = lbann.Reduction(is_not_pad, mode='sum',name='num_not_pad')

# Cross entropy loss 
label_tokens = lbann.Slice(
	label_tokens,
        slice_points=range(sequence_length+1),
	name='label_tokens',
    )

label_tokens = [lbann.Identity(label_tokens) for _ in range(sequence_length)]

loss = []


for i in range(sequence_length):
	label = lbann.OneHot(label_tokens[i], size=config.vocab_size)
	label = lbann.Reshape(label, dims=[config.vocab_size])
	pred = lbann.Reshape(preds[i], dims=[config.vocab_size]) 
	loss.append(lbann.CrossEntropy(pred, label))


loss = lbann.Concatenation(loss)


# Average cross entropy over non-pad tokens
loss_scales = lbann.SafeDivide(
	is_not_pad,
        lbann.Tessellate(num_not_pad, hint_layer=is_not_pad),
        name = 'loss_scale',
    )
loss = lbann.Multiply(loss, loss_scales)


obj = lbann.Reduction(loss, mode='sum',name='loss_red')

metrics = [lbann.Metric(obj, name="loss")]


###########
# Callbacks
###########

callbacks = [lbann.CallbackPrint(),
             lbann.CallbackTimer(),]


callbacks.append(
	lbann.CallbackDumpOutputs(
		batch_interval=782,
		execution_modes='train', 
		directory=os.path.join(work_dir, 'train_input'),
		layers='input_strings')
    )


callbacks.append(
	lbann.CallbackDumpOutputs(
		batch_interval=782,
		execution_modes='train', 
		directory=os.path.join(work_dir, 'train_output'),
		layers='pred ')
    )

callbacks.append(
	lbann.CallbackDumpOutputs(
		batch_interval=50,
		execution_modes='test', 
		directory=os.path.join(work_dir, 'test_input'),
		layers='input_strings')
    )

callbacks.append(
	lbann.CallbackDumpOutputs(
		batch_interval=50,
		execution_modes='test', 
		directory=os.path.join(work_dir, 'test_output'),
		layers='pred')
    )


callbacks.append(
	lbann.CallbackDumpWeights(
		directory=os.path.join(work_dir, 'weights'),
		epoch_interval=1,
	)
    )


model = lbann.Model(
    lbann_params.epochs,
    layers=lbann.traverse_layer_graph(input_),
    objective_function=obj,
    metrics=metrics,
    callbacks=callbacks,
)

# Setup trainer, optimizer, data_reader
trainer = lbann.Trainer(
    mini_batch_size=lbann_params.mini_batch_size,
    num_parallel_readers=1,
)
optimizer = lbann.Adam(
    learn_rate=0.0001,
    beta1=0.9,
    beta2=0.98,
    eps=1e-8,
)
data_reader = make_data_reader()

# Launch LBANN
kwargs = lbann.contrib.args.get_scheduler_kwargs(lbann_params)
kwargs["environment"] = {}
lbann.contrib.launcher.run(
    trainer,
    model,
    data_reader,
    optimizer,
    work_dir=work_dir,
    job_name=lbann_params.job_name,
    lbann_args=["--num_io_threads=1"],
    batch_job=lbann_params.batch_job,
    **kwargs,
)
