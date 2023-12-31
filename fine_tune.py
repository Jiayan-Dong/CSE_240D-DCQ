#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""This is an example application for compressing image classification models.

The application borrows its main flow code from torchvision's ImageNet classification
training sample application (https://github.com/pytorch/examples/tree/master/imagenet).
We tried to keep it similar, in order to make it familiar and easy to understand.

Integrating compression is very simple: simply add invocations of the appropriate
compression_scheduler callbacks, for each stage in the training.  The training skeleton
looks like the pseudo code below.  The boiler-plate Pytorch classification training
is speckled with invocations of CompressionScheduler.

For each epoch:
    compression_scheduler.on_epoch_begin(epoch)
    train()
    validate()
    save_checkpoint()
    compression_scheduler.on_epoch_end(epoch)

train():
    For each training step:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input)
        loss = criterion(output, target)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)


This exmple application can be used with torchvision's ImageNet image classification
models, or with the provided sample models:

- ResNet for CIFAR: https://github.com/junyuseu/pytorch-cifar-models
- MobileNet for ImageNet: https://github.com/marvis/pytorch-mobilenet
"""

import math
import argparse
import time
import os
import sys
import random
import traceback
import logging
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchnet.meter as tnt
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, r'src'))
try:
    import distiller
except ImportError:
    sys.path.append(module_path)
    import distiller
sys.path.append(os.path.join(script_dir, r'src'))
import apputils
from distiller.data_loggers import *
import distiller.quantization as quantization
from models import ALL_MODEL_NAMES, create_model

# Logger handle
msglogger = None

def float_range(val_str):
    val = float(val_str)
    if val < 0 or val >= 1:
        raise argparse.ArgumentTypeError('Must be >= 0 and < 1 (received {0})'.format(val_str))
    return val

parser = argparse.ArgumentParser(description='Distiller image classification model compression')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=ALL_MODEL_NAMES,
                    help='model architecture: ' +
                    ' | '.join(ALL_MODEL_NAMES) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--partial_epoch', default=1.0, type=float, metavar='N',
                    help='number of partial epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--fpresume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')                    
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--act-stats', dest='activation_stats', choices=["train", "valid", "test"], default=None,
                    help='collect activation statistics (WARNING: this slows down training)')
parser.add_argument('--masks-sparsity', dest='masks_sparsity', action='store_true', default=False,
                    help='print masks sparsity table at end of each epoch')
parser.add_argument('--param-hist', dest='log_params_histograms', action='store_true', default=False,
                    help='log the paramter tensors histograms to file (WARNING: this can use significant disk space)')
SUMMARY_CHOICES = ['sparsity', 'compute', 'model', 'modules', 'png', 'png_w_params', 'onnx']
parser.add_argument('--summary', type=str, choices=SUMMARY_CHOICES,
                    help='print a summary of the model, and exit - options: ' +
                    ' | '.join(SUMMARY_CHOICES))
parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                    help='configuration file for pruning the model (default is to use hard-coded schedule)')
parser.add_argument('--fpcompress', dest='fpcompress', type=str, nargs='?', action='store',
                    help='configuration file for restore the fp model (default is to use hard-coded schedule)')
parser.add_argument('--sense', dest='sensitivity', choices=['element', 'filter', 'channel'],
                    help='test the sensitivity of layers to pruning')
parser.add_argument('--sense-range', dest='sensitivity_range', type=float, nargs=3, default=[0.0, 0.95, 0.05],
                    help='an optional paramaeter for sensitivity testing providing the range of sparsities to test.\n'
                    'This is equaivalent to creating sensitivities = np.arange(start, stop, step)')
parser.add_argument('--extras', default=None, type=str,
                    help='file with extra configuration information')
parser.add_argument('--deterministic', '--det', action='store_true',
                    help='Ensure deterministic execution for re-producible results.')
parser.add_argument('--gpus', metavar='DEV_ID', default=None,
                    help='Comma-separated list of GPU device IDs to be used (default is to use all available devices)')
parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
parser.add_argument('--out-dir', '-o', dest='output_dir', default='logs', help='Path to dump logs and checkpoints')
parser.add_argument('--validation-size', '--vs', type=float_range, default=0.1,
                    help='Portion of training dataset to set aside for validation')
parser.add_argument('--adc', dest='ADC', action='store_true', help='temp HACK')
parser.add_argument('--adc-params', dest='ADC_params', default=None, help='temp HACK')
parser.add_argument('--confusion', dest='display_confusion', default=False, action='store_true',
                    help='Display the confusion matrix')
parser.add_argument('--num-best-scores', dest='num_best_scores', default=1, type=int,
                    help='number of best scores to track and report (default: 1)')
parser.add_argument('--load-serialized', dest='load_serialized', action='store_true', default=False,
                    help='Load a model without DataParallel wrapping it')

str_to_quant_mode_map = {'sym': quantization.LinearQuantMode.SYMMETRIC,
                         'asym_s': quantization.LinearQuantMode.ASYMMETRIC_SIGNED,
                         'asym_u': quantization.LinearQuantMode.ASYMMETRIC_UNSIGNED}


def linear_quant_mode_str(val_str):
    try:
        return str_to_quant_mode_map[val_str]
    except KeyError:
        raise argparse.ArgumentError('Must be one of {0} (received {1})'.format(list(str_to_quant_mode_map.keys()),
                                                                                val_str))


quant_group = parser.add_argument_group('Arguments controlling quantization at evaluation time'
                                        '("post-training quantization)')
quant_group.add_argument('--quantize-eval', '--qe', action='store_true',
                         help='Apply linear quantization to model before evaluation. Applicable only if'
                              '--evaluate is also set')
quant_group.add_argument('--qe-bits-acts', '--qeba', type=int, default=8, metavar='NUM_BITS',
                         help='Number of bits for quantization of activations')
quant_group.add_argument('--qe-bits-wts', '--qebw', type=int, default=8, metavar='NUM_BITS',
                         help='Number of bits for quantization of weights')
quant_group.add_argument('--qe-bits-accum', type=int, default=32, metavar='NUM_BITS',
                         help='Number of bits for quantization of the accumulator')
quant_group.add_argument('--qe-clip-acts', '--qeca', action='store_true',
                         help='Enable clipping of activations using min/max values averaging over batch')
quant_group.add_argument('--qe-no-clip-layers', '--qencl', type=str, nargs='+', metavar='LAYER_NAME', default=[],
                         help='List of layer names for which not to clip activations. Applicable only if '
                              '--qe-clip-acts is also set')
quant_group.add_argument('--qe-per-channel', '--qepc', action='store_true',
                         help='Enable per-channel quantization of weights (per output channel)')

distiller.knowledge_distillation.add_distillation_args(parser, ALL_MODEL_NAMES, True)


def check_pytorch_version():
    if torch.__version__ < '0.4.0':
        print("\nNOTICE:")
        print("The Distiller \'master\' branch now requires at least PyTorch version 0.4.0 due to "
              "PyTorch API changes which are not backward-compatible.\n"
              "Please install PyTorch 0.4.0 or its derivative.\n"
              "If you are using a virtual environment, do not forget to update it:\n"
              "  1. Deactivate the old environment\n"
              "  2. Install the new environment\n"
              "  3. Activate the new environment")
        exit(1)


def create_activation_stats_collectors(model, collection_phase):
    """Create objects that collect activation statistics.

    This is a utility function that creates two collectors:
    1. Fine-grade sparsity levels of the activations
    2. L1-magnitude of each of the activation channels

    Args:
        model - the model on which we want to collect statistics
        phase - the statistics collection phase which is either "train" (for training),
                or "valid" (for validation)

    WARNING! Enabling activation statsitics collection will significantly slow down training!
    """
    class missingdict(dict):
        """This is a little trick to prevent KeyError"""
        def __missing__(self, key):
            return None  # note, does *not* set self[key] - we don't want defaultdict's behavior

    distiller.utils.assign_layer_fq_names(model)

    activations_collectors = {"train": missingdict(), "valid": missingdict(), "test": missingdict()}
    if collection_phase is None:
        return activations_collectors
    collectors = missingdict({
        "sparsity":      SummaryActivationStatsCollector(model, "sparsity",
                                                         lambda t: 100 * distiller.utils.sparsity(t)),
        "l1_channels":   SummaryActivationStatsCollector(model, "l1_channels",
                                                         distiller.utils.activation_channels_l1),
        "apoz_channels": SummaryActivationStatsCollector(model, "apoz_channels",
                                                         distiller.utils.activation_channels_apoz),
        "records":       RecordsActivationStatsCollector(model, classes=[torch.nn.Conv2d])})
    activations_collectors[collection_phase] = collectors
    return activations_collectors


def save_collectors_data(collectors, directory):
    """Utility function that saves all activation statistics to Excel workbooks
    """
    for name, collector in collectors.items():
        workbook = os.path.join(directory, name)
        msglogger.info("Generating {}".format(workbook))
        collector.to_xlsx(workbook)

class DorefaParamsBinarizationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, inplace=False):
        if inplace:
            ctx.mark_dirty(input)
        E = input.abs().mean()
        output = torch.where(input == 0, torch.ones_like(input), torch.sign(input)) * E # Binary
        # output = torch.where(input == 0, torch.ones_like(input), torch.sign(input))
        # output = torch.sign(input) * E # Ternary
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def main():
    global msglogger
    check_pytorch_version()
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir)
    print(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir)
    print(msglogger)
    # assert False

    # Log various details about the execution environment.  It is sometimes useful
    # to refer to past experiment executions and this information may be useful.
    apputils.log_execution_env_state(sys.argv, gitroot=module_path)
    msglogger.debug("Distiller: %s", distiller.__version__)

    start_epoch = 0
    best_epochs = [distiller.MutableNamedTuple({'epoch': 0, 'top1': 0, 'sparsity': 0})
                   for i in range(args.num_best_scores)]

    if args.deterministic:
        # Experiment reproducibility is sometimes important.  Pete Warden expounded about this
        # in his blog: https://petewarden.com/2018/03/19/the-machine-learning-reproducibility-crisis/
        # In Pytorch, support for deterministic execution is still a bit clunky.
        if args.workers > 1:
            msglogger.error('ERROR: Setting --deterministic requires setting --workers/-j to 0 or 1')
            exit(1)
        # Use a well-known seed, for repeatability of experiments
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        cudnn.deterministic = True
    else:
        # This issue: https://github.com/pytorch/pytorch/issues/3659
        # Implies that cudnn.benchmark should respect cudnn.deterministic, but empirically we see that
        # results are not re-produced when benchmark is set. So enabling only if deterministic mode disabled.
        cudnn.benchmark = True

    if args.gpus is not None:
        try:
            args.gpus = [int(s) for s in args.gpus.split(',')]
        except ValueError:
            msglogger.error('ERROR: Argument --gpus must be a comma-separated list of integers only')
            exit(1)
        available_gpus = torch.cuda.device_count()
        for dev_id in args.gpus:
            if dev_id >= available_gpus:
                msglogger.error('ERROR: GPU device ID {0} requested, but only {1} devices available'
                                .format(dev_id, available_gpus))
                exit(1)
        # Set default device in case the first one on the list != 0
        torch.cuda.set_device(args.gpus[0])

    # Infer the dataset from the model name
    if 'cifar' in args.arch:
        args.dataset = 'cifar10'
    elif 'mnist' in args.arch:
        args.dataset = 'mnist'
    elif 'svhn' in args.arch:
        args.dataset = 'svhn'
    else:
        args.dataset = 'imagenet'

    args.num_classes = 10 if args.dataset == 'cifar10' or args.dataset == 'svhn' else 1000

    # Create the model
    model = create_model(args.pretrained, args.dataset, args.arch,
                         parallel=not args.load_serialized, device_ids=args.gpus)
    original_model = create_model(args.pretrained, args.dataset, args.arch,
                         parallel=not args.load_serialized, device_ids=args.gpus)
    compression_scheduler = None
    # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
    # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
    tflogger = TensorBoardLogger(msglogger.logdir)
    pylogger = PythonLogger(msglogger)

    # We can optionally resume from a checkpoint
    if args.resume:
        print(args)
        model, compression_scheduler, start_epoch = apputils.load_checkpoint(
            model, chkpt_file=args.resume)
        #original_model, original_compression_scheduler, start_epoch = apputils.load_checkpoint(
        if args.fpresume:
            original_model, original_compression_scheduler, original_start_epoch = apputils.load_checkpoint(
                original_model, chkpt_file=args.fpresume)
            original_model.cuda()
        model.cuda()
        

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    """ edit optimizer"""
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    original_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, original_model.parameters()), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    msglogger.info('Optimizer Type: %s', type(optimizer))
    msglogger.info('Optimizer Args: %s', optimizer.defaults)

    # This sample application can be invoked to produce various summary reports.
    if args.summary:
        return summarize_model(model, args.dataset, which_summary=args.summary)

    # Load the datasets: the dataset to load is inferred from the model name passed
    # in args.arch.  The default dataset is ImageNet, but if args.arch contains the
    # substring "_cifar", then cifar10 is used.
    train_loader, val_loader, test_loader, _ = apputils.load_data(
        args.dataset, os.path.expanduser(args.data), args.batch_size,
        args.workers, args.validation_size, args.deterministic)
    msglogger.info('Dataset sizes:\n\ttraining=%d\n\tvalidation=%d\n\ttest=%d',
                   len(train_loader.sampler), len(val_loader.sampler), len(test_loader.sampler))

    activations_collectors = create_activation_stats_collectors(model, collection_phase=args.activation_stats)

    if args.evaluate:
        return evaluate_model(model, criterion, test_loader, pylogger, activations_collectors, args)

    if args.compress:
        # The main use-case for this sample application is CNN compression. Compression
        # requires a compression schedule configuration file in YAML.
        compression_scheduler = distiller.file_config(model, optimizer, args.compress, compression_scheduler)
        if args.fpresume:
            original_compression_scheduler = distiller.file_config(original_model, original_optimizer, args.fpcompress, original_compression_scheduler)

        # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
        model.cuda()
        #original_model.cuda()
    elif compression_scheduler is None:
        compression_scheduler = distiller.CompressionScheduler(model)

    args.kd_policy = None
    if args.kd_teacher:
        teacher = create_model(args.kd_pretrained, args.dataset, args.kd_teacher, device_ids=args.gpus)
        if args.kd_resume:
            teacher, _, _ = apputils.load_checkpoint(teacher, chkpt_file=args.kd_resume)
        dlw = distiller.DistillationLossWeights(args.kd_distill_wt, args.kd_student_wt, args.kd_teacher_wt)
        args.kd_policy = distiller.KnowledgeDistillationPolicy(model, teacher, args.kd_temp, dlw)
        compression_scheduler.add_policy(args.kd_policy, starting_epoch=args.kd_start_epoch, ending_epoch=args.epochs,
                                         frequency=1)

        msglogger.info('\nStudent-Teacher knowledge distillation enabled:')
        msglogger.info('\tTeacher Model: %s', args.kd_teacher)
        msglogger.info('\tTemperature: %s', args.kd_temp)
        msglogger.info('\tLoss Weights (distillation | student | teacher): %s',
                       ' | '.join(['{:.2f}'.format(val) for val in dlw]))
        msglogger.info('\tStarting from Epoch: %s', args.kd_start_epoch)

    print("=======================================(Model)======")
    print(model)
    
    print("=======================================(original_model)======")
    print(original_model)
        
    with collectors_context(activations_collectors["valid"]) as collectors:
            top1, top5, vloss = validate(val_loader, model, criterion, [pylogger], args, 0)
            save_collectors_data(collectors, msglogger.logdir)
    print('model pre-training:', top1)
    with collectors_context(activations_collectors["valid"]) as collectors:
            top1, top5, vloss = validate(val_loader, original_model, criterion, [pylogger], args, 0)
            save_collectors_data(collectors, msglogger.logdir)
    print('original_model pre-training:', top1)
    print('kd ratio:', args.kd_distill_wt)
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # This is the main training loop.
        msglogger.info('\n')
        
        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch)
        
        if args.evaluate:
            return evaluate_model(model, criterion, test_loader, pylogger, activations_collectors, args)

        xp = []
        for p in model.parameters():
            xp.append(p.clone())
        
        # Train for one epoch
        with collectors_context(activations_collectors["train"]) as collectors:
            train(train_loader, model, original_model, criterion, optimizer, epoch, compression_scheduler,
                  loggers=[tflogger, pylogger], args=args)

        with collectors_context(activations_collectors["valid"]) as collectors:
            top1, top5, vloss = validate(val_loader, model, criterion, [pylogger], args, epoch)
            distiller.log_activation_statsitics(epoch, "valid", loggers=[tflogger],
                                                collector=collectors["sparsity"])
            save_collectors_data(collectors, msglogger.logdir)

        stats = ('Peformance/Validation/',
                 OrderedDict([('Loss', vloss),
                              ('Top1', top1),
                              ('Top5', top5)]))
        distiller.log_training_progress(stats, None, epoch, steps_completed=0, total_steps=1, log_freq=1,
                                        loggers=[tflogger])

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch, optimizer)

        # Update the list of top scores achieved so far, and save the checkpoint
        is_best = top1 > best_epochs[-1].top1
        if top1 > best_epochs[0].top1:
            best_epochs[0].epoch = epoch
            best_epochs[0].top1 = top1
            # Keep best_epochs sorted such that best_epochs[0] is the lowest top1 in the best_epochs list
            best_epochs = sorted(best_epochs, key=lambda score: score.top1)
        for score in reversed(best_epochs):
            if score.top1 > 0:
                msglogger.info('==> Best Top1: %.3f on Epoch: %d', score.top1, score.epoch)
        apputils.save_checkpoint(epoch, args.arch, model, optimizer, compression_scheduler,
                                 best_epochs[-1].top1, is_best, args.name, msglogger.logdir)
        

    # Finally run results on the test set
    evaluate_model(model, criterion, test_loader, pylogger, activations_collectors, args)
    #test(test_loader, model, criterion, [pylogger], activations_collectors, args=args)


OVERALL_LOSS_KEY = 'Overall Loss'
OBJECTIVE_LOSS_KEY = 'Objective Loss'


def train(train_loader, model, original_model, criterion, optimizer, epoch,
          compression_scheduler, loggers, args):
    """Training loop for one epoch."""
    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))
    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()

    total_samples = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)
    msglogger.info('Training epoch: %d samples (%d per mini-batch)', total_samples, batch_size)
    epoch_frac = args.partial_epoch
    steps_per_frac_epoch = math.ceil((total_samples*epoch_frac) / batch_size)

    # Switch to train mode
    model.train()
    original_model.eval()
    end = time.time()

    for train_step, (inputs, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.add(time.time() - end)
        inputs, target = inputs.to('cuda'), target.to('cuda')

        if train_step == steps_per_frac_epoch:
            break
        # Execute the forward phase, compute the output and measure loss
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)

        if args.kd_policy is None:
            torch.set_printoptions(precision=10)
            """ current ternary (quantized) model"""
            output = model(inputs)
            
            """ original full-precision model"""
            fp_output = original_model(inputs).detach()
            
            """ different losses """
            # loss = criterion(output, target)
            ratio = 0.0
            loss1 = criterion(output, target)
            new_criterion = nn.MSELoss()
            loss2 = new_criterion(torch.nn.functional.softmax(output, dim=1), torch.nn.functional.softmax(fp_output, dim=1))
            loss = loss1 * (1 - ratio) + loss2 * ratio
            
            """ Measure accuracy and record loss """
            classerr.add(output.data, target)
        else:
            output = args.kd_policy.forward(inputs)
        
        losses[OBJECTIVE_LOSS_KEY].add(loss.item())

        if compression_scheduler:
            # Before running the backward phase, we allow the scheduler to modify the loss
            # (e.g. add regularization loss)
            agg_loss = compression_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, loss,
                                                                  optimizer=optimizer, return_loss_components=True)
            loss = agg_loss.overall_loss
            losses[OVERALL_LOSS_KEY].add(loss.item())
            for lc in agg_loss.loss_components:
                if lc.name not in losses:
                    losses[lc.name] = tnt.AverageValueMeter()
                losses[lc.name].add(lc.value.item())

        # Compute the gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)
        
        # measure elapsed time
        batch_time.add(time.time() - end)
        steps_completed = (train_step+1)

        if steps_completed % args.print_freq == 0:
            # Log some statistics
            errs = OrderedDict()
            errs['Top1'] = classerr.value(1)
            errs['Top5'] = classerr.value(5)
            

            stats_dict = OrderedDict()
            for loss_name, meter in losses.items():
                stats_dict[loss_name] = meter.mean
            stats_dict.update(errs)
            stats_dict['LR'] = optimizer.param_groups[0]['lr']
            stats_dict['Time'] = batch_time.mean
            stats = ('Peformance/Training/', stats_dict)

            params = model.named_parameters() if args.log_params_histograms else None
            distiller.log_training_progress(stats,
                                            params,
                                            epoch, steps_completed,
                                            steps_per_epoch, args.print_freq,
                                            loggers)
        end = time.time()

def validate(val_loader, model, criterion, loggers, args, epoch=-1):
    """Model validation"""
    if epoch > -1:
        msglogger.info('--- validate (epoch=%d)-----------', epoch)
    else:
        msglogger.info('--- validate ---------------------')
    return _validate(val_loader, model, criterion, loggers, args, epoch)


def test(test_loader, model, criterion, loggers, activations_collectors, args):
    """Model Test"""
    msglogger.info('--- test ---------------------')

    with collectors_context(activations_collectors["test"]) as collectors:
        top1, top5, lossses = _validate(test_loader, model, criterion, loggers, args)
        distiller.log_activation_statsitics(-1, "test", loggers, collector=collectors['sparsity'])
        save_collectors_data(collectors, msglogger.logdir)
    return top1, top5, lossses


def _validate(data_loader, model, criterion, loggers, args, epoch=-1):
    """Execute the validation/test loop."""
    losses = {'objective_loss': tnt.AverageValueMeter()}
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1, 5))

    batch_time = tnt.AverageValueMeter()
    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    if args.display_confusion:
        confusion = tnt.ConfusionMeter(args.num_classes)
    total_steps = total_samples / batch_size
    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

    # Switch to evaluation mode
    model.eval()

    end = time.time()
    for validation_step, (inputs, target) in enumerate(data_loader):
        with torch.no_grad():
            inputs, target = inputs.to('cuda'), target.to('cuda')
            # compute output from model
            output = model(inputs)

            # compute loss
            loss = criterion(output, target)
            # measure accuracy and record loss
            losses['objective_loss'].add(loss.item())
            classerr.add(output.data, target)
            if args.display_confusion:
                confusion.add(output.data, target)

            # measure elapsed time
            batch_time.add(time.time() - end)
            end = time.time()

            steps_completed = (validation_step+1)
            if steps_completed % args.print_freq == 0:
                stats = ('',
                        OrderedDict([('Loss', losses['objective_loss'].mean),
                                        ('Top1', classerr.value(1)),
                                        ('Top5', classerr.value(5))]))


                distiller.log_training_progress(stats, None, epoch, steps_completed,
                                                total_steps, args.print_freq, loggers)

    msglogger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                    classerr.value()[0], classerr.value()[1], losses['objective_loss'].mean)

    if args.display_confusion:
        msglogger.info('==> Confusion:\n%s\n', str(confusion.value()))
    return classerr.value(1), classerr.value(5), losses['objective_loss'].mean

def evaluate_model(model, criterion, test_loader, loggers, activations_collectors, args):
    # This sample application can be invoked to evaluate the accuracy of your model on
    # the test dataset.
    # You can optionally quantize the model to 8-bit integer before evaluation.
    # For example:
    # python3 compress_classifier.py --arch resnet20_cifar  ../data.cifar10 -p=50 --resume=checkpoint.pth.tar --evaluate

    if not isinstance(loggers, list):
        loggers = [loggers]

    if args.quantize_eval:
        model.cpu()
        quantizer = quantization.PostTrainLinearQuantizer(model, args.qe_bits_acts, args.qe_bits_wts,
                                                          args.qe_bits_accum, args.qe_mode, args.qe_clip_acts,
                                                          args.qe_no_clip_layers, args.qe_per_channel)
        quantizer.prepare_model()
        model.cuda()

    top1, _, _ = test(test_loader, model, criterion, loggers, activations_collectors, args=args)

    if args.quantize_eval:
        checkpoint_name = 'quantized'
        apputils.save_checkpoint(0, args.arch, model, optimizer=None, best_top1=top1,
                                 name='_'.join([args.name, checkpoint_name]) if args.name else checkpoint_name,
                                 dir=msglogger.logdir)


def summarize_model(model, dataset, which_summary):
    if which_summary.startswith('png'):
        apputils.draw_img_classifier_to_file(model, 'model.png', dataset, which_summary == 'png_w_params')
    elif which_summary == 'onnx':
        apputils.export_img_classifier_to_onnx(model, 'model.onnx', dataset)
    else:
        distiller.model_summary(model, which_summary, dataset)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None:
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))
