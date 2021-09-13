import argparse
import lbann
import lbann.models
import lbann.models.resnet
import lbann.contrib.args
import lbann.contrib.models.wide_resnet
import lbann.contrib.launcher
import data.imagenet

# Command-line arguments
desc = ('Construct and run ResNet on ImageNet-1K data. '
        'Running the experiment is only supported on LC systems.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='lbann_resnet', type=str,
    help='scheduler job name (default: lbann_resnet)')
parser.add_argument(
    '--resnet', action='store', default=50, type=int,
    choices=(18, 34, 50, 101, 152),
    help='ResNet variant (default: 50)')
parser.add_argument(
    '--width', action='store', default=2, type=float,
    help='Wide ResNet width factor (default: 2)')
parser.add_argument(
    '--block-type', action='store', default=None, type=str,
    choices=('basic', 'bottleneck'),
    help='ResNet block type')
parser.add_argument(
    '--blocks', action='store', default=None, type=str,
    help='ResNet block counts (comma-separated list)')
parser.add_argument(
    '--block-channels', action='store', default=None, type=str,
    help='Internal channels in each ResNet block (comma-separated list)')
parser.add_argument(
    '--bn-statistics-group-size', action='store', default=1, type=int,
    help=('Group size for aggregating batch normalization statistics '
          '(default: 1)'))
parser.add_argument(
    '--warmup', action='store_true', help='use a linear warmup')
parser.add_argument(
    '--mini-batch-size', action='store', default=32, type=int,
    help='mini-batch size (default: 16)', metavar='NUM') #256
parser.add_argument(
    '--num-epochs', action='store', default=90, type=int,
    help='number of epochs (default: 90)', metavar='NUM')
parser.add_argument(
    '--num-classes', action='store', default=1000, type=int,
    help='number of ImageNet classes (default: 1000)', metavar='NUM')
parser.add_argument(
    '--random-seed', action='store', default=0, type=int,
    help='random seed for LBANN RNGs', metavar='NUM')
lbann.contrib.args.add_optimizer_arguments(parser, default_learning_rate=0.1)
args = parser.parse_args()

# Due to a data reader limitation, the actual model realization must be
# hardcoded to 1000 labels for ImageNet.
imagenet_labels = 1000

# Choose ResNet variant
resnet_variant_dict = {18: lbann.models.ResNet18,
                       34: lbann.models.ResNet34,
                       50: lbann.models.ResNet50,
                       101: lbann.models.ResNet101,
                       152: lbann.models.ResNet152}
wide_resnet_variant_dict = {50: lbann.contrib.models.wide_resnet.WideResNet50_2}
block_variant_dict = {
    'basic': lbann.models.resnet.BasicBlock,
    'bottleneck': lbann.models.resnet.BottleneckBlock
}

if (any([args.block_type, args.blocks, args.block_channels])
    and not all([args.block_type, args.blocks, args.block_channels])):
    raise RuntimeError('Must specify all of --block-type, --blocks, --block-channels')
if args.block_type and args.blocks and args.block_channels:
    # Build custom ResNet.
    resnet = lbann.models.ResNet(
        block_variant_dict[args.block_type],
        imagenet_labels,
        list(map(int, args.blocks.split(','))),
        list(map(int, args.block_channels.split(','))),
        zero_init_residual=True,
        bn_statistics_group_size=args.bn_statistics_group_size,
        name='custom_resnet',
        width=args.width)
elif args.width == 1:
    # Vanilla ResNet.
    resnet = resnet_variant_dict[args.resnet](
        imagenet_labels,
        bn_statistics_group_size=args.bn_statistics_group_size)
elif args.width == 2 and args.resnet == 50:
    # Use pre-defined WRN-50-2.
    resnet = wide_resnet_variant_dict[args.resnet](
        imagenet_labels,
        bn_statistics_group_size=args.bn_statistics_group_size)
else:
    # Some other Wide ResNet.
    resnet = resnet_variant_dict[args.resnet](
        imagenet_labels,
        bn_statistics_group_size=args.bn_statistics_group_size,
        width=args.width)

# Construct layer graph
input_ = lbann.Input(target_mode='classification')
images = lbann.Identity(input_)
labels = lbann.Identity(input_)


#Aug
mag_rot = '0'
rot_weights = lbann.Weights(initializer=lbann.ValueInitializer(values=mag_rot), name='rot_weights', optimizer=lbann.NoOptimizer())
rot = lbann.WeightsLayer(dims='1', weights=rot_weights, name='rot', device='CPU')


mag_shear = '0 0'
shear_weights = lbann.Weights(initializer=lbann.ValueInitializer(values=mag_shear),name='shear_weights',optimizer=lbann.NoOptimizer())
shear = lbann.WeightsLayer(dims='2', weights=shear_weights, name='shear')
  

mag_trans = '0 0'
trans_weights = lbann.Weights(initializer=lbann.ValueInitializer(values=mag_trans),name='trans_weights',optimizer=lbann.NoOptimizer())
trans = lbann.WeightsLayer(dims='2', weights=trans_weights, name='trans')


images = lbann.CompositeImageTransformation(images,rot,shear,trans,device='CPU')


preds = resnet(images)
probs = lbann.Softmax(preds)
cross_entropy = lbann.CrossEntropy(probs, labels)
top1 = lbann.CategoricalAccuracy(probs, labels)
top5 = lbann.TopKCategoricalAccuracy(probs, labels, k=5)
layers = list(lbann.traverse_layer_graph(input_))

# Setup tensor core operations (just to demonstrate enum usage)
tensor_ops_mode = lbann.ConvTensorOpsMode.NO_TENSOR_OPS
for l in layers:
    if type(l) == lbann.Convolution:
        l.conv_tensor_op_mode=tensor_ops_mode

# Setup objective function
l2_reg_weights = set()
for l in layers:
    if type(l) == lbann.Convolution or type(l) == lbann.FullyConnected:
        l2_reg_weights.update(l.weights)
l2_reg = lbann.L2WeightRegularization(weights=l2_reg_weights, scale=1e-4)
obj = lbann.ObjectiveFunction([cross_entropy, l2_reg])

# Setup model
metrics = [lbann.Metric(top1, name='accuracy', unit='%'),
           lbann.Metric(top5, name='top-5 accuracy', unit='%')]

callbacks = [lbann.CallbackPrint(),
             lbann.CallbackTimer(),
             lbann.CallbackDropFixedLearningRate(
                 drop_epoch=[30, 60, 80], amt=0.1)]

callbacks.append(lbann.CallbackPerturbWeights(output_name='rot_weights',batch_interval=400,lower=-5,upper=5,scale=5,perturb_probability=0.5))
callbacks.append(lbann.CallbackPerturbWeights(output_name='shear_weights',batch_interval=400,lower=-0.15,upper=0.15,scale=0.15,perturb_probability=0.5))
callbacks.append(lbann.CallbackPerturbWeights(output_name='trans_weights',batch_interval=400,lower=-5,upper=5,scale=5,perturb_probability=0.5))

if args.warmup:
    callbacks.append(
        lbann.CallbackLinearGrowthLearningRate(
            target=0.1 * args.mini_batch_size / 256, num_epochs=5))
model = lbann.Model(args.num_epochs,
                    layers=layers,
                    objective_function=obj,
                    metrics=metrics,
                    callbacks=callbacks)

# Setup optimizer
opt = lbann.contrib.args.create_optimizer(args)

# Setup data reader
data_reader = data.imagenet.make_data_reader(num_classes=args.num_classes)

# Setup trainer
#trainer = lbann.Trainer(mini_batch_size=args.mini_batch_size, random_seed=args.random_seed)

RPE = lbann.RandomPairwiseExchange
SGD = lbann.BatchedIterativeOptimizer
metalearning = RPE(
                     metric_strategies={'accuracy': RPE.MetricStrategy.HIGHER_IS_BETTER})
ltfb = lbann.LTFB("ltfb",
                   metalearning=metalearning,
                   local_algo=SGD("local sgd",
                                   num_iterations=400),
                   metalearning_steps=40)

trainer = lbann.Trainer(mini_batch_size=32,
                        training_algo=ltfb)

# Run experiment
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
nodes_args=32
lbann.contrib.launcher.run(trainer, model, data_reader, opt,nodes=nodes_args,
                           job_name=args.job_name,
                           lbann_args=f"--procs_per_trainer=4",
                           **kwargs)
