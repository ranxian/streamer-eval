# eval.py - Evaluate the performance of streamer
from termcolor import colored
import subprocess
from threading import Thread
from subprocess import call
from subprocess import Popen
from argparse import ArgumentParser
from metric_collector import MetricCollector
import numpy as np
import time
import os
import matplotlib
import numpy as np

matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plot
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import LinearLocator


LOCAL_DIR = "/Users/xianran/Code/TX1/streamer/build"
LOCAL_HOST_DIR = "localhost:/Users/xianran/Code/TX1/streamer/build"
NUC_HOST_DIR = "sc02:/home/rxian/tx1dnn-caffe-opencl-gen9/build"
TEGRA_HOST_DIR = "tegra-1:/home/ubuntu/Code/streamer/build"

# Default values
DEFAULT_NETWORK = "GoogleNet"
DEFAULT_CAMERA = "AMCBLACK1"
DEFAULT_ITERATION = 1000
DEFAULT_ENCODER = 'omxh264enc'
DEFAULT_DECODER = 'omxh264dec'
DEFAULT_DEVICE_NUMBER = 0  # GPU #0
DEFAULT_HOST_DIR = TEGRA_HOST_DIR

DEBUG = False

RESULT_DIR_BASE = 'results'
# Heter experiment
HETER_DIR = RESULT_DIR_BASE + '/heter'
HETER_EXP_ENCODERS = ['x264enc', 'omxh264enc', 'vaapih264enc']
HETER_EXP_DECODERS = ['avdec_h264', 'omxh264dec', 'vaapidecode']
HETER_FILE = "heter.tsv"
HETER_EXP_CPU_ITERATION = 20
HETER_EXP_GPU_ITERATION = DEFAULT_ITERATION
HETER_EXP_HOST_DIRS = [TEGRA_HOST_DIR, NUC_HOST_DIR]

# Imagenet experiment
IMAGENET_DIR = RESULT_DIR_BASE + '/imagenet'
IMAGENET_NETWORK_NAMES = [
    'AlexNet', 'GoogleNet', 'VGG', 'ResNet', 'FCN', 'CPM']
IMAGENET_EXP_ITERATION = 200


def print_error(msg):
  """Print message in alert color"""
  print colored(msg, 'red')


def sync_codebase(host_dir):
  ssh_command = "rsync -avh --delete --exclude=.git/* --exclude=build/* --exclude=.idea/* --exclude=config/* %s/.. %s/.." % (
      LOCAL_DIR, host_dir)
  if not DEBUG:
    ssh_command += " > /dev/null"
  if call(ssh_command, shell=True) != 0:
    print_error("Can't sync codebase with: " + host_dir)
    exit(-1)


def remote_execute(host_dir, *cmd):
  host, streamer_dir = host_dir.split(":")
  ssh_command = "ssh %s 'zsh -l -c \"(cd %s; %s)\"'" % (
      host, streamer_dir, ' '.join(cmd))

  # if DEBUG:
  print ssh_command

  p = Popen(
      ssh_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  output = p.communicate()

  if p.returncode != 0:
    print_error("Error executing: %s on %s" % (host_dir, cmd))
    print output[0]
    print output[1]
    exit(-1)

  if DEBUG:
    print output[0]
    print output[1]

  return output


def build_streamer(host_dir):
  remote_execute(host_dir, "make -j")


def clean_dir(dir):
  call("rm -r %s" % (dir), shell=True)


def GET_RESULT_FILENAME(paths):
  return '/'.join(paths)


def write_result(filename, metrics):
  dirname = os.path.dirname(filename)
  if not os.path.exists(dirname):
    os.makedirs(dirname)

  with open(filename, 'a') as f:
    f.write('\t'.join([str(m) for m in metrics]) + '\n')


def run_experiment(host_dir,
                   network=DEFAULT_NETWORK,
                   camera=DEFAULT_CAMERA,
                   on_tegra=False,
                   on_nuc=False,
                   encoder=DEFAULT_ENCODER,
                   decoder=DEFAULT_DECODER,
                   device_number=DEFAULT_DEVICE_NUMBER,
                   iteration=DEFAULT_ITERATION):

  def grab_fps(result, pattern):
    for line in result.split("\n"):
      if pattern in line and "fps" in line:
        return float(line.split(" ")[-1])
    return 0.0

  host, _ = host_dir.split(":")
  collector = MetricCollector(host, on_tegra=on_tegra)
  collector.start()

  log_level = 2
  if DEBUG:
    log_level = 0

  output, _ = remote_execute(host_dir,
                             "GLOG_minloglevel=%d apps/benchmark" % (
                                 log_level),
                             '--experiment', 'classification',
                             '--camera', camera,
                             '--net', network,
                             '--verbose', 'true',
                             '--iter', str(iteration),
                             "--encoder", encoder,
                             "--decoder", decoder,
                             "--device", str(device_number))
  collector.stop()
  cpu = collector.get_cpu_usage()
  mem = collector.get_mem_usage()
  fps = grab_fps(output, "classifier")

  return cpu, mem, fps


def heter_eval():
  clean_dir(HETER_DIR)

  for host_dir in HETER_EXP_HOST_DIRS:
    on_tegra = host_dir == TEGRA_HOST_DIR
    on_nuc = host_dir == NUC_HOST_DIR
    platform = 'tegra' if on_tegra else 'nuc'
    iteration_cpu = HETER_EXP_CPU_ITERATION if on_tegra else 200
    iteration_gpu = HETER_EXP_GPU_ITERATION if on_tegra else 200

    sync_codebase(host_dir)
    build_streamer(host_dir)

    remote_execute(host_dir, "sleep 1")

    # Fix GPU, vary hardware encoder/decoders
    for idx, encoder in enumerate(HETER_EXP_ENCODERS):
      if on_tegra and encoder == "vaapih264enc":
        continue
      if on_nuc and encoder == "omxh264enc":
        continue

      decoder = HETER_EXP_DECODERS[idx]
      result_file = GET_RESULT_FILENAME(
          [HETER_DIR, platform, encoder+"+"+decoder, HETER_FILE])
      result = run_experiment(
          host_dir,
          on_tegra=on_tegra,
          on_nuc=on_nuc,
          encoder=encoder,
          decoder=decoder,
          iteration=iteration_gpu)
      write_result(result_file, result)

    # Fix hardware encoder/decoder, varying CPU mode or device mode
    hardware_decoder = \
        HETER_EXP_DECODERS[1] if on_tegra else HETER_EXP_DECODERS[2]
    hardware_encoder = \
        HETER_EXP_ENCODERS[1] if on_tegra else HETER_EXP_ENCODERS[2]
    result_file = GET_RESULT_FILENAME([HETER_DIR, platform, 'cpu', HETER_FILE])
    result = run_experiment(host_dir, on_tegra=True,
                            encoder=hardware_encoder,
                            decoder=hardware_decoder,
                            device_number=-1,
                            iteration=iteration_cpu)
    write_result(result_file, result)

    result_file = GET_RESULT_FILENAME([HETER_DIR, platform, 'gpu', HETER_FILE])
    result = run_experiment(host_dir, on_tegra=True,
                            encoder=hardware_encoder,
                            decoder=hardware_decoder,
                            device_number=0,
                            iteration=iteration_gpu)
    write_result(result_file, result)


def imagenet_eval():
  pass


def heter_plot():
  pass


def imagenet_plot():
  pass


if __name__ == "__main__":
  parser = ArgumentParser("Evaluate performance of streamer")
  parser.add_argument(
      "--heter_eval", help="Run heterogeneous experiment", action='store_true')
  parser.add_argument(
      "--imagenet_eval", help="Run imagenet experiment", action='store_true')
  parser.add_argument(
      "--heter_plot", help="Plot heterogeneous experiment", action='store_true')
  parser.add_argument(
      "--imagenet_plot", help="Plot imagenet experiment", action='store_true')
  parser.add_argument(
      "--debug", help="Enable debugging", action='store_true')

  args = parser.parse_args()

  if args.debug:
    DEBUG = True

  if args.heter_eval:
    heter_eval()

  if args.imagenet_eval:
    imagenet_eval()

  if args.heter_plot:
    heter_plot()

  if args.imagenet_plot:
    imagenet_plot()
