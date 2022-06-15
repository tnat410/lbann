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

from lbann.models import RoBERTa, RoBERTaMLM


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
    default=100,
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
    'Roberta_mlm/exps',
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
        #log_probs = x
        #log = lbann.Identity(log_probs)

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
        return loss#,log


# ----------------------------------------------
# Build and Run Model
# ----------------------------------------------
with open("./config.json") as f:
    config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
config.input_shape = (1,57)
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

robertamlm = RoBERTaMLM(config,load_weights=config.load_weights)
output = robertamlm(input_masked)

preds = lbann.Identity(output,name='pred')


preds = lbann.ChannelwiseSoftmax(preds, name='pred_sm')
preds = lbann.Slice(preds, axis=1, slice_points=range(sequence_length+1),name='slice_pred')
preds = [lbann.Identity(preds) for _ in range(sequence_length)]


########
# Loss
########

# Count number of masked tokens
label_tokens = lbann.Identity(input_label)

non_masked = lbann.Constant(value=ignore_index, num_neurons=sequence_length,name='non_masked')
is_masked = lbann.NotEqual(label_tokens, non_masked ,name='is_masked')
num_masked = lbann.Reduction(is_masked, mode='sum',name='num_masked')

# Cross entropy loss 
label_tokens = lbann.Slice(
	label_tokens,
        slice_points=range(sequence_length+1),
	name='label_tokens',
    )

label_tokens = [lbann.Identity(label_tokens) for _ in range(sequence_length)]

loss = []

loss_func = CrossEntropyLoss(vocab_size, data_layout="model_parallel")
for i in range(sequence_length):
	obj_loss = loss_func(preds[i], label_tokens[i])
	loss.append(obj_loss)


loss = lbann.Concatenation(loss)

# Average cross entropy over masked tokens
loss_scales = lbann.Divide(
	is_masked,
        lbann.Tessellate(num_masked, hint_layer=is_masked),
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
		layers='pred')
    )

callbacks.append(
	lbann.CallbackDumpOutputs(
		batch_interval=98,
		execution_modes='test', 
		directory=os.path.join(work_dir, 'test_input'),
		layers='input_strings')
    )

callbacks.append(
	lbann.CallbackDumpOutputs(
		batch_interval=98,
		execution_modes='test', 
		directory=os.path.join(work_dir, 'test_output'),
		layers='pred')
    )

callbacks.append(
	lbann.CallbackDumpWeights(
		directory=os.path.join(work_dir, 'weights'),
		epoch_interval=2,
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
