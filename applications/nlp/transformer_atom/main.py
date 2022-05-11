"""Driver script for training Transformer example."""
import argparse
import datetime
import math
import os
import os.path
import sys

import lbann
import lbann.contrib.args

# Local imports
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
import train
#import evaluate
import utils.paths

# ----------------------------------------------
# Options
# ----------------------------------------------

# Command-line arguments
parser = argparse.ArgumentParser()
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='transformer_atom', type=str,
    help='job name', metavar='NAME')
parser.add_argument(
    '--mini-batch-size', action='store', default=256, type=int,
    help='mini-batch size (default: 256)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=30, type=int,
    help='number of epochs (default: 30)', metavar='NUM')
parser.add_argument(
    '--num-attention-heads', action='store', default=4, type=int,
    help='number of parallel attention layers (default: 4)', metavar='NUM')
parser.add_argument(
    '--embed-dim', action='store', default=256, type=int,
    help='embedding space dimensions (default: 256)', metavar='NUM')
args = parser.parse_args()

# Hard-coded options
label_smoothing = 0.1

# ----------------------------------------------
# Work directory
# ----------------------------------------------

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
work_dir = os.path.join(
    utils.paths.root_dir(),
    'experiments',
    f'{timestamp}_{args.job_name}',
)
os.makedirs(work_dir, exist_ok=True)

# ----------------------------------------------
# Train
# ----------------------------------------------

# Create batch script
trainer_params = {
    'mini_batch_size': args.mini_batch_size,
}
model_params = {
    'num_epochs': args.num_epochs,
    'embed_dim': args.embed_dim,
    'num_heads': args.num_attention_heads,
    'label_smoothing': label_smoothing,
}
script_params = lbann.contrib.args.get_scheduler_kwargs(args)
script_params['work_dir'] = work_dir
script_params['job_name'] = args.job_name
train_script = train.make_batch_script(
    trainer_params=trainer_params,
    model_params=model_params,
    script_params=script_params,
)
weights_prefix = os.path.join(
    work_dir,
    'weights',
    f'model0-epoch{args.num_epochs-1}',
)
train_script.add_command(
    f'# python3 {utils.paths.root_dir()}/transformer/evaluate.py {weights_prefix}'
)
train_script.run(overwrite=True)

# ----------------------------------------------
# Evaluate
# ----------------------------------------------
#evaluate.evaluate_transformer(weights_prefix)
