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


# Host and paths of streamer deployment
LOCAL_DIR = "/Users/xianran/Code/TX1/streamer/build"
LOCAL_HOST_DIR = "localhost:/Users/xianran/Code/TX1/streamer/build"
NUC_HOST_DIR = "sc02:/home/rxian/tx1dnn-caffe-opencl-gen9/build"
TEGRA_HOST_DIR = "tegra-1:/home/ubuntu/Code/streamer/build"

# Plot configurations
OPT_FONT_NAME = 'Helvetica'
OPT_GRAPH_HEIGHT = 150
OPT_GRAPH_WIDTH = 400

NUM_COLORS = 5
COLOR_MAP = ('#418259', '#bd5632', '#e1a94c', '#7d6c5b', '#efefef')

OPT_COLORS = COLOR_MAP

OPT_GRID_COLOR = 'gray'
OPT_LEGEND_SHADOW = False
OPT_MARKERS = (
    ['o', 's', 'v', "^", "h", "v", ">", "x", "d", "<", "|", "", "|", "_"])
OPT_PATTERNS = (["////", "o", "\\\\", ".", "\\\\\\"])

OPT_STACK_COLORS = ('#2b3742', '#c9b385', '#610606', '#1f1501')
OPT_LINE_STYLES = ('-', ':', '--', '-.')

# SET FONT

OPT_LABEL_WEIGHT = 'bold'
OPT_LINE_COLORS = COLOR_MAP
OPT_LINE_WIDTH = 6.0
OPT_MARKER_SIZE = 10.0

AXIS_LINEWIDTH = 1.3
BAR_LINEWIDTH = 1.2

LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 14
TINY_FONT_SIZE = 10
LEGEND_FONT_SIZE = 18

SMALL_LABEL_FONT_SIZE = 10
SMALL_LEGEND_FONT_SIZE = 10

XAXIS_MIN = 0.25
XAXIS_MAX = 3.75

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['font.family'] = OPT_FONT_NAME
matplotlib.rcParams['pdf.use14corefonts'] = True

LABEL_FP = FontProperties(style='normal', size=LABEL_FONT_SIZE, weight='bold')
TICK_FP = FontProperties(style='normal', size=TICK_FONT_SIZE)
TINY_FP = FontProperties(style='normal', size=TINY_FONT_SIZE)
LEGEND_FP = FontProperties(
    style='normal', size=LEGEND_FONT_SIZE, weight='bold')

SMALL_LABEL_FP = FontProperties(
    style='normal', size=SMALL_LABEL_FONT_SIZE, weight='bold')
SMALL_LEGEND_FP = FontProperties(
    style='normal', size=SMALL_LEGEND_FONT_SIZE, weight='bold')

YAXIS_TICKS = 3
YAXIS_ROUND = 1000.0

END_TO_END_EXPERIMENT = 'endtoend'
NNINFER_EXPERIMENT = 'nninfer'

# Default values
DEFAULT_NETWORK = "GoogleNet"
DEFAULT_CAMERA = "AMCBLACK1"
DEFAULT_DURATION = 60
DEFAULT_ENCODER = 'omxh264enc'
DEFAULT_DECODER = 'omxh264dec'
DEFAULT_DEVICE_NUMBER = 0  # GPU #0
DEFAULT_HOST_DIR = TEGRA_HOST_DIR
DEFAULT_EXPERIMENT = END_TO_END_EXPERIMENT
DEFAULT_STORE = False

DEBUG = False

RESULT_DIR_BASE = 'results'

# Heter experiment
HETER_DIR = RESULT_DIR_BASE + '/heter'
HETER_EXP_ENCODERS = ['x264enc', 'omxh264enc', 'vaapih264enc']
HETER_EXP_DECODERS = ['avdec_h264', 'omxh264dec', 'vaapidecode']
HETER_FILE = "heter.tsv"
HETER_EXP_HOST_DIRS = [NUC_HOST_DIR, TEGRA_HOST_DIR]
HETER_FPS_KEYWORDS = ['classifier']

# Imagenet experiment
IMAGENET_DIR = RESULT_DIR_BASE + '/imagenet'
IMAGENET_NETWORK_NAMES = [
    'AlexNet', 'SqueezeNet', 'GoogleNet', 'VGG', 'ResNet', 'FCN']
IMAGENET_FPS_KEYWORDS = ['processor']
IMAGENET_FILE = "imagenet.tsv"


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
  remote_execute(host_dir, "cmake ..")
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


def read_result(filename):
  with open(filename, 'r') as f:
    return f.readline().strip().split("\t")


def run_experiment(host_dir,
                   fps_keywords,
                   network=DEFAULT_NETWORK,
                   camera=DEFAULT_CAMERA,
                   on_tegra=False,
                   on_nuc=False,
                   encoder=DEFAULT_ENCODER,
                   decoder=DEFAULT_DECODER,
                   device_number=DEFAULT_DEVICE_NUMBER,
                   duration=DEFAULT_DURATION,
                   experiment=DEFAULT_EXPERIMENT,
                   store=DEFAULT_STORE):

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
                             '--experiment', experiment,
                             '--camera', camera,
                             '--net', network,
                             '--verbose', 'true',
                             '--time', str(duration),
                             "--encoder", encoder,
                             "--decoder", decoder,
                             "--device", str(device_number),
                             "--store", str(store))
  collector.stop()
  results = []
  results.append(collector.get_cpu_usage())
  results.append(collector.get_mem_usage())

  fps = []
  for keyword in fps_keywords:
    results.append(grab_fps(output, keyword))

  return results


def heter_eval():
  clean_dir(HETER_DIR)

  for host_dir in HETER_EXP_HOST_DIRS:
    on_tegra = host_dir == TEGRA_HOST_DIR
    on_nuc = host_dir == NUC_HOST_DIR
    platform = 'tegra' if on_tegra else 'nuc'

    sync_codebase(host_dir)
    build_streamer(host_dir)

    remote_execute(host_dir, "sleep 1")

    for device, backend in zip([-1, 0], ["cpu", "gpu"]):
      for encoder, decoder in zip(HETER_EXP_ENCODERS, HETER_EXP_DECODERS):
        if on_tegra and encoder == "vaapih264enc":
          continue
        if on_nuc and encoder == "omxh264enc":
          continue
        use_hardware_endecoder = (encoder != 'x264enc')
        decoder_type = "hde" if use_hardware_endecoder else "sde"

        result_file = GET_RESULT_FILENAME(
            [HETER_DIR, platform, backend+"-"+decoder_type, HETER_FILE])
        result = run_experiment(
            host_dir,
            on_tegra=on_tegra,
            on_nuc=on_nuc,
            encoder=encoder,
            decoder=decoder,
            device_number=device,
            store=True,
            fps_keywords=HETER_FPS_KEYWORDS)
        write_result(result_file, result)


def imagenet_eval():
  clean_dir(IMAGENET_DIR)

  on_tegra = True
  platform = 'tegra'

  sync_codebase(TEGRA_HOST_DIR)
  build_streamer(TEGRA_HOST_DIR)

  remote_execute(TEGRA_HOST_DIR, 'sleep 1')

  for network in IMAGENET_NETWORK_NAMES:
    result_file = GET_RESULT_FILENAME([IMAGENET_DIR, network, IMAGENET_FILE])
    result = run_experiment(
        TEGRA_HOST_DIR,
        on_tegra=on_tegra,
        experiment=NNINFER_EXPERIMENT,
        fps_keywords=IMAGENET_FPS_KEYWORDS,
        network=network)
    write_result(result_file, result)

####### PLOT ########


def saveGraph(fig, output, width, height):
  OUTPUT_DIR = "figures"
  if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

  output = "./figures/" + output

  size = fig.get_size_inches()
  dpi = fig.get_dpi()

  new_size = (width / float(dpi), height / float(dpi))
  fig.set_size_inches(new_size)
  new_size = fig.get_size_inches()
  new_dpi = fig.get_dpi()

  if output.endswith(".pdf"):
    pp = PdfPages(output)
  else:
    pp = output
  fig.savefig(pp, format=output.split(".")[-1], bbox_inches='tight')

  if output.endswith(".pdf"):
    pp.close()

  print "Saved graph to " + output


def heter_plot():
  for platform in ['nuc', 'tegra']:
    for backend in ['cpu', 'gpu']:
      for enc_dec in ['software', 'hardware']:
        result = read_result(GET_RESULT_FILENAME([HETER_DIR,
                                                  platform,
                                                  backend+"-"+enc_dec,
                                                  HETER_FILE]))
        print platform, backend, enc_dec, result


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
