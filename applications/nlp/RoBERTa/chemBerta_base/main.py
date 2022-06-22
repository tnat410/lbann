from types import SimpleNamespace
import argparse
import datetime
import os
import sys
import json
import numpy as np
import math

import lbann
import lbann.contrib.args
import lbann.contrib.launcher

from lbann.models import RoBERTa, RoBERTaMLM
from lbann.util import make_iterable
import lbann.modules

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
    'Roberta_zinc_base/exps',
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
# LayerNorm
# ----------------------------------------------

class LayerNorm(lbann.modules.Module):
    """See https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html"""

    global_count = 0  # Static counter, used for default names

    def __init__(
            self,
            normalized_shape,
            name=None,
            weight=[],	
            bias=[],		
    ):
        super().__init__()
        LayerNorm.global_count += 1
        self.normalized_shape = make_iterable(normalized_shape)
        self.name = (name
                     if name
                     else f'layernorm{LayerNorm.global_count}')

        if(weight):
                self.weight = weight
                self.bias = bias
        else:
                # Initialize weights
                self.weight = lbann.Weights(
		    initializer=lbann.ConstantInitializer(value=1),
		    name=f'{self.name}_weight',
                )
                self.bias = lbann.Weights(
		    initializer=lbann.ConstantInitializer(value=0),
		    name=f'{self.name}_bias',
                )

    def forward(self, x):

        # Normalization
        x = lbann.InstanceNorm(x)

        # Affine transform
        s = lbann.WeightsLayer(
            weights=self.weight,
            dims=[1] + list(make_iterable(self.normalized_shape)),
        )
        s = lbann.Tessellate(s, hint_layer=x)
        b = lbann.WeightsLayer(
            weights=self.bias,
            dims=[1] + list(make_iterable(self.normalized_shape)),
        )
        b = lbann.Tessellate(b, hint_layer=x)
        x = lbann.Add(lbann.Multiply(s,x), b)
        return x

# ----------------------------------------------
# Loss
# ----------------------------------------------
class CrossEntropyLoss(lbann.modules.Module):
    """Cross-entropy loss for classification.

    Given an input vector x, weight matrix W, and label y:

      L = -log( softmax(W*x) * onehot(y) )

    Args:
      num_classes (int): Number of class.
      weights (lbann.Weights): Matrix with dimensions of
        num_classes x input_size. Each row is an embedding vector
        for the corresponding class.
      data_layout (str): Data layout of fully-connected layer.

    """

    def __init__(
        self,
        num_classes,
        weights=[],
        data_layout="data_parallel",
    ):
        self.num_classes = num_classes
        self.data_layout = data_layout
        self.fc = lbann.modules.FullyConnectedModule(
            self.num_classes,
            weights=weights,
            bias=False,
            activation=lbann.LogSoftmax,
            name="class_fc",
            data_layout=self.data_layout,
        )

    def forward(self, x, label):
        """Compute cross-entropy loss.

        Args:
          x (lbann.Layer): Input vector.
          label (lbann.Layer): Label. Should have one entry, which
            will be cast to an integer.

        Returns:
          lbann.Layer: Loss function value.

        """
        log_probs = self.fc(x)
        label_onehot = lbann.OneHot(
            label,
            size=self.num_classes,
            data_layout=self.data_layout,
        )
        loss = lbann.Multiply(
            log_probs,
            label_onehot,
            data_layout=self.data_layout,
        )
        loss = lbann.Reduction(
            loss,
            mode="sum",
            data_layout=self.data_layout,
        )
        loss = lbann.Negative(loss, data_layout=self.data_layout)
        return loss


# ----------------------------------------------
# Gelu
# ----------------------------------------------
def Gelu(x):
    x_erf = lbann.Erf(lbann.Scale(x, constant=(1 / math.sqrt(2))))
    return lbann.Multiply(
        x, lbann.Scale(lbann.AddConstant(x_erf, constant=1), constant=0.5)
    )

# ----------------------------------------------
# load weight
# ----------------------------------------------
def _load_pretrained_weights(
    *fn,
    file_dir=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pretrained_weights"
    ),
    load_weights=True,
):
    if not load_weights:
        return []

    # Use custom directory for loading weights
    if isinstance(load_weights, str):
        file_dir = load_weights

    weights = []
    for f in fn:
        w_file = os.path.join(file_dir, f + ".npy")
        if not os.path.isfile(w_file):
            raise ValueError(f"Pretrained weight file does not exist: {w_file}")
        weights.append(lbann.Weights(initializer=lbann.NumpyInitializer(file=w_file)))

    if len(weights) == 1:
        weights = weights[0]
    return weights

# ----------------------------------------------
# Build and Run Model
# ----------------------------------------------
with open("./config.json") as f:
    config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
config.input_shape = (1,sequence_length)
config.load_weights = os.path.abspath('./pretrained_weights')
#config.load_weights = False

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



roberta = RoBERTa(config,add_pooling_layer=False,load_weights=config.load_weights)
output = roberta(input_masked)



# Reconstruct decoder input

#Dense
output = lbann.Reshape(output, dims=(sequence_length,config.hidden_size),name='sample')

preds = lbann.ChannelwiseFullyConnected(
        output,
        weights=_load_pretrained_weights(
                ".".join(("lm_head", "dense.weight")),
                load_weights=config.load_weights,
            ),
        output_channel_dims=[config.hidden_size],
        bias=False,
        name='linear',
)

#Norm
norms = LayerNorm(config.hidden_size,
		weight=_load_pretrained_weights(
			".".join(("lm_head", "layer_norm.weight")),
			load_weights=config.load_weights,
            	),
		bias=_load_pretrained_weights(
			".".join(("lm_head", "layer_norm.bias")),
			load_weights=config.load_weights,
            	),
	)

preds = norms(preds)

#Gelu
preds = Gelu(preds)


#Decoder
preds = lbann.ChannelwiseFullyConnected(
        preds,
        weights=_load_pretrained_weights(
                ".".join(("lm_head.decoder", "weight")),
                ".".join(("lm_head.decoder", "bias")),
                load_weights=config.load_weights,
            ),
        output_channel_dims=[config.vocab_size],
        bias=True,
        name='decoder',
)


preds = lbann.ChannelwiseSoftmax(preds, name='pred_sm')
preds = lbann.Slice(preds, axis=0, slice_points=range(sequence_length+1),name='slice_pred')
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
	label = lbann.OneHot(label_tokens[i], size=vocab_size)
	label = lbann.Reshape(label, dims=[vocab_size])
	pred = lbann.Reshape(preds[i], dims=[vocab_size])
	loss.append(lbann.CrossEntropy(pred, label))


loss = lbann.Concatenation(loss)


# Average cross entropy over non-pad tokens
loss_scales = lbann.SafeDivide(
	is_not_pad,
        lbann.Tessellate(num_not_pad, hint_layer=is_not_pad),
    )
loss = lbann.Multiply(loss, loss_scales)

obj = lbann.Reduction(loss, mode='sum',name='loss_red')

metrics = [lbann.Metric(obj, name="loss")]


###########
# Callbacks
###########

callbacks = [lbann.CallbackPrint(),
             lbann.CallbackTimer(),
             lbann.CallbackCheckNaN(),]


callbacks.append(
	lbann.CallbackDumpOutputs(
		batch_interval=782*10,
		execution_modes='train', 
		directory=os.path.join(work_dir, 'train_input'),
		layers='input_strings input_masked')
    )


callbacks.append(
	lbann.CallbackDumpOutputs(
		batch_interval=782*10,

		execution_modes='train', 
		directory=os.path.join(work_dir, 'train_output'),
		layers='pred pred_sm')
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
		layers='pred pred_sm')
    )


callbacks.append(
	lbann.CallbackDumpWeights(
		directory=os.path.join(work_dir, 'weights'),
		epoch_interval=10,
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
    learn_rate=0.00005,
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
