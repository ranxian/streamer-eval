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

###### Configurations ######
# Host and paths of streamer deployment
LOCAL_DIR = "/Users/xianran/Code/TX1/streamer/build"
LOCAL_HOST_DIR = "localhost:/Users/xianran/Code/TX1/streamer/build"
NUC_HOST_DIR = "sc02:/home/rxian/tx1dnn-caffe-opencl-gen9/build"
TEGRA_HOST_DIR = "tegra-1:/home/ubuntu/Code/streamer/build"
TEGRA_BVLC_CAFFE_DIR = 'tegra-1:/home/ubuntu/Code/streamer-bvlc-caffe/build'
TEGRA_FP16_CAFFE_DIR = 'tegra-1:/home/ubuntu/Code/streamer-fp16-caffe/build'
TEGRA_MXNET_DIR = 'tegra-1:/home/ubuntu/Code/streamer-mxnet/build'
TEGRA_TENSORRT_DIR = 'tegra-1:/home/ubuntu/Code/streamer-tensorrt/build'
TEGRA_TENSORFLOW_DIR = 'tegra-1:/home/ubuntu/Code/streamer-tensorflow/build'
NUC_MKL_CAFFE_DIR = 'sc02:/home/rxian/tx1dnn-intelcaffe-mkl-cpu/build'
TEGRA_CONFIG_DIR = "/home/ubuntu/Code/config"
NUC_CONFIG_DIR = "/home/rxian/config"

HOST_DIC = {
    "local": LOCAL_HOST_DIR,
    "nuc": NUC_HOST_DIR,
    "tegra": TEGRA_HOST_DIR,
    "tegra-bvlc": TEGRA_BVLC_CAFFE_DIR,
    "tegra-fp16": TEGRA_FP16_CAFFE_DIR,
    "tegra-mxnet": TEGRA_MXNET_DIR,
    "tegra-gie": TEGRA_TENSORRT_DIR,
    "nuc-mkl": NUC_MKL_CAFFE_DIR,
}

END_TO_END_EXPERIMENT = 'endtoend'
NNINFER_EXPERIMENT = 'nninfer'

# Default values
DEFAULT_NETWORK = "GoogleNet"
DEFAULT_CAMERA = "AMCBLACK1"
DEFAULT_DURATION = 120
DEFAULT_ENCODER = 'omxh264enc'
DEFAULT_DECODER = 'omxh264dec'
DEFAULT_DEVICE_NUMBER = 0  # GPU #0
DEFAULT_HOST_DIR = TEGRA_HOST_DIR
DEFAULT_EXPERIMENT = END_TO_END_EXPERIMENT
DEFAULT_STORE = False
DEFAULT_CONFIG_DIR = './config'
DEFAULT_BATCH_SIZE = 1
DEFAULT_USE_FP16 = False

DEBUG = False

RESULT_DIR_BASE = 'results'
# Heter experiment
HETER_DIR = RESULT_DIR_BASE + '/heterogeneous'
HETER_EXP_ENCODERS = ['x264enc', 'omxh264enc', 'vaapih264enc']
HETER_EXP_DECODERS = ['avdec_h264', 'omxh264dec', 'vaapidecode']
HETER_EXP_HOST_DIRS = [NUC_HOST_DIR, TEGRA_HOST_DIR]
HETER_EXP_FPS_KEYWORDS = ['classifier']
HETER_FILE = "heter.tsv"

# Imagenet experiment
IMAGENET_DIR = RESULT_DIR_BASE + '/imagenet'
IMAGENET_EXP_NETWORK_NAMES = [
    'AlexNet', 'SqueezeNet', 'GoogleNet', 'VGG', 'ResNet', 'FCN']
IMAGENET_EXP_FPS_KEYWORDS = ['processor']
IMAGENET_FILE = "imagenet.tsv"

# Framework experiment
FRAMEWORK_DIR = RESULT_DIR_BASE + '/framework'
FRAME_WORK_EXP_BATCH_SIZES = [1, 2, 4, 8]
FRAMEWORK_EXP_FRAMEWORKS = {
    'caffe-bvlc': {
        'host_dir': TEGRA_BVLC_CAFFE_DIR,
        'config_dir': TEGRA_CONFIG_DIR,
        'networks': ['AlexNet', 'GoogleNet', 'ResNet']
    },
    'mxnet': {
        'host_dir': TEGRA_MXNET_DIR,
        'config_dir': TEGRA_CONFIG_DIR,
        'networks': ['InceptionBN', 'InceptionV3', 'ResNet-MXNet-50']
    },
    'tensorrt-fp16': {
        'host_dir': TEGRA_TENSORRT_DIR,
        'config_dir': TEGRA_CONFIG_DIR,
        'networks': ['AlexNetGIE', 'GoogleNetGIE']
    },
    'tensorrt': {
        'host_dir': TEGRA_TENSORRT_DIR,
        'config_dir': TEGRA_CONFIG_DIR,
        'networks': ['AlexNetGIE', 'GoogleNetGIE', 'ResNetGIE']
    },
    # # 'tensorflow': {}
    'caffe-fp16': {
        'host_dir': TEGRA_FP16_CAFFE_DIR,
        'config_dir': TEGRA_CONFIG_DIR,
        'networks': ['AlexNetFP16', 'GoogleNetFP16'],  # 'SqueezeNet1.0FP16']
    },
    'caffe-mkl': {
        'host_dir': NUC_MKL_CAFFE_DIR,
        'config_dir': NUC_CONFIG_DIR,
        'networks': ['AlexNet', "GoogleNet"]
    },
}
FRAMEWORK_EXP_FPS_KEYWORDS = ['processor']
FRAMEWORK_FILE = 'framework.tsv'

# Scalability experiment
SCALABILITY_DIR = RESULT_DIR_BASE + '/scalability'
SCALABILITY_EXP_FPS_KEYWORDS = ['classifier']
SCALABILITY_EXP_BATCH_SIZES = [1, 2, 4, 7]
SCALABILITY_EXP_CAMERA_BASENAME = 'AMCBLACK'
SCALABILITY_EXP_NETWORKS = ['AlexNet', 'GoogleNet']
SCALABILITY_FILE = 'scalability.tsv'


def print_error(msg):
  """Print message in alert color"""
  print colored(msg, 'red')


def sync_codebase(host_dir):
  ssh_command = "rsync -avh --delete --exclude=.git/* --exclude=build/* --exclude=.idea/* --exclude=./config/* %s/.. %s/.." % (
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
  remote_execute(host_dir, "make -j8")


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
  results = []
  with open(filename, 'r') as f:
    for line in f:
      results.append(line.strip().split("\t"))
  return results


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
                   store=DEFAULT_STORE,
                   config_dir=DEFAULT_CONFIG_DIR,
                   batch=DEFAULT_BATCH_SIZE,
                   use_fp16=DEFAULT_USE_FP16):

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
                             "--store", str(store),
                             "--config_dir", config_dir,
                             "--batch", str(batch),
                             "--fp16", str(use_fp16))
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
            fps_keywords=HETER_EXP_FPS_KEYWORDS)
        write_result(result_file, result)


def imagenet_eval():
  clean_dir(IMAGENET_DIR)

  on_tegra = True
  platform = 'tegra'

  sync_codebase(TEGRA_HOST_DIR)
  build_streamer(TEGRA_HOST_DIR)

  remote_execute(TEGRA_HOST_DIR, 'sleep 1')

  for network in IMAGENET_EXP_NETWORK_NAMES:
    result_file = GET_RESULT_FILENAME([IMAGENET_DIR, network, IMAGENET_FILE])
    result = run_experiment(
        TEGRA_HOST_DIR,
        on_tegra=on_tegra,
        experiment=NNINFER_EXPERIMENT,
        fps_keywords=IMAGENET_EXP_FPS_KEYWORDS,
        network=network)
    write_result(result_file, result)


def framework_eval():
  clean_dir(FRAMEWORK_DIR)

  for name in FRAMEWORK_EXP_FRAMEWORKS.keys():
    framework = FRAMEWORK_EXP_FRAMEWORKS[name]
    sync_codebase(framework['host_dir'])
    build_streamer(framework['host_dir'])
    remote_execute(framework['host_dir'], 'sleep 1')

    on_nuc = 'mkl' in name
    on_tegra = not on_nuc
    use_fp16 = "fp16" in name

    for network in framework['networks']:
      for batch_size in FRAME_WORK_EXP_BATCH_SIZES:
        result_file = GET_RESULT_FILENAME(
            [FRAMEWORK_DIR, name, network, FRAMEWORK_FILE])
        result = run_experiment(
            framework['host_dir'],
            FRAMEWORK_EXP_FPS_KEYWORDS,
            on_tegra=on_tegra,
            on_nuc=on_nuc,
            experiment=NNINFER_EXPERIMENT,
            network=network,
            config_dir=framework['config_dir'],
            device_number=0 if on_tegra else -1,
            batch=batch_size,
            use_fp16=use_fp16)
        write_result(result_file, result)


def scalability_eval():
  clean_dir(SCALABILITY_DIR)

  sync_codebase(TEGRA_HOST_DIR)
  build_streamer(TEGRA_HOST_DIR)
  remote_execute(TEGRA_HOST_DIR, 'sleep 1')

  on_tegra = True

  def get_camera_names(batch_size):
    camera_names = []
    for i in xrange(1, batch_size+1):
      camera_name = SCALABILITY_EXP_CAMERA_BASENAME + str(i)
      camera_names.append(camera_name)
    return ','.join(camera_names)

  for network in SCALABILITY_EXP_NETWORKS:
    for batch_size in SCALABILITY_EXP_BATCH_SIZES:
      camera_names = get_camera_names(batch_size)
      result_file = GET_RESULT_FILENAME(
          [SCALABILITY_DIR, network, SCALABILITY_FILE])
      result = run_experiment(TEGRA_HOST_DIR,
                              SCALABILITY_EXP_FPS_KEYWORDS,
                              on_tegra=on_tegra,
                              experiment=END_TO_END_EXPERIMENT,
                              network=network,
                              config_dir=TEGRA_CONFIG_DIR,
                              camera=camera_names,
                              batch=batch_size,
                              store=True)
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


def MEM_STR(mem):
  return "%.2f MB" % (float(mem)/1024.0)


def heter_plot():
  HEADERS = ['PLATFORM', 'DEVICE', 'CODECS', 'CPU', 'MEM', 'FPS']
  print '\t'.join(HEADERS)
  for platform in ['nuc', 'tegra']:
    for backend in ['cpu', 'gpu']:
      for enc_dec in ['sde', 'hde']:
        codecs = 'software' if enc_dec == 'sde' else 'hardware'
        result = read_result(GET_RESULT_FILENAME([HETER_DIR,
                                                  platform,
                                                  backend+"-"+enc_dec,
                                                  HETER_FILE]))[0]
        result[1] = MEM_STR(result[1])
        print '\t'.join([platform, backend, codecs, '\t'.join(result)])


def imagenet_plot():
  HEADERS = ['NETWORK', 'CPU', 'MEM', 'FPS']
  print '\t'.join(HEADERS)
  for network in IMAGENET_EXP_NETWORK_NAMES:
    result = read_result(
        GET_RESULT_FILENAME([IMAGENET_DIR, network, IMAGENET_FILE]))[0]
    result[1] = MEM_STR(result[1])
    print '\t'.join([network, '\t'.join(result)])


def framework_plot():
  HEADERS = ['FRAMEWORK', 'NETWORK', 'BATCH', 'CPU', 'MEM', 'FPS']
  print '\t'.join(HEADERS)
  for name in FRAMEWORK_EXP_FRAMEWORKS.keys():
    framework = FRAMEWORK_EXP_FRAMEWORKS[name]
    for network in framework['networks']:
      results = read_result(
          GET_RESULT_FILENAME([FRAMEWORK_DIR, name, network, FRAMEWORK_FILE]))
      for result, batch_size in zip(results, [1, 2, 4, 8]):
        result[1] = MEM_STR(result[1])
        result[2] = str(float(result[2]) * batch_size)
        print '\t'.join([name, network, str(batch_size), '\t'.join(result)])
    print ''


def scalability_plot():
  HEADERS = ['NETWORK', 'BATCH', 'CPU', 'MEM', 'FPS']
  print '\t'.join(HEADERS)

  for network in SCALABILITY_EXP_NETWORKS:
    results = read_result(
        GET_RESULT_FILENAME([SCALABILITY_DIR, network, SCALABILITY_FILE]))
    for result, batch_size in zip(results, [1, 2, 4, 8]):
      result[1] = MEM_STR(result[1])
      result[2] = str(float(result[2]) * batch_size)
      print '\t'.join([network, str(batch_size), '\t'.join(result)])


def sync_streamer(host_name):
  """Sync and build streamer on a remote machine"""
  host_dir = HOST_DIC[host_name]
  sync_codebase(host_dir)
  build_streamer(host_dir)


def print_summary():
  print 'HETEROGENEOUS EXPERIMENT'
  heter_plot()
  print ''
  print 'IMAGENET EXPERIMENT'
  imagenet_plot()
  print ''
  print 'FRAMEWORK EXPERIMENT'
  framework_plot()
  print ''
  print 'SCALABILITY EXPERIMENT'
  scalability_plot()


if __name__ == "__main__":
  parser = ArgumentParser("Evaluate performance of streamer")
  parser.add_argument(
      "--heter_eval", help="Run heterogeneous experiment", action='store_true')
  parser.add_argument(
      "--imagenet_eval", help="Run imagenet experiment", action='store_true')
  parser.add_argument(
      "--framework_eval", help="Run multiple framework experiment", action='store_true')
  parser.add_argument(
      "--scalability_eval", help="Run scalability experiment", action='store_true')
  parser.add_argument(
      "--sync", help="Sync the codebase with a remote host: nuc, nuc-mkl, tegra, tegra-fp16-caffe, tegra-gie, tegra-mxnet")

  parser.add_argument(
      "--heter_plot", help="Plot heterogeneous experiment", action='store_true')
  parser.add_argument(
      "--imagenet_plot", help="Plot imagenet experiment", action='store_true')
  parser.add_argument(
      "--framework_plot", help="Plot framework experiment", action='store_true')
  parser.add_argument(
      "--scalability_plot", help="Plot scalability experiment", action='store_true')
  parser.add_argument(
      "--summary", help="Print result summary", action='store_true')
  parser.add_argument(
      "--debug", help="Enable debugging", action='store_true')

  args = parser.parse_args()

  if args.debug:
    DEBUG = True

  if args.heter_eval:
    heter_eval()

  if args.imagenet_eval:
    imagenet_eval()

  if args.framework_eval:
    framework_eval()

  if args.scalability_eval:
    scalability_eval()

  if args.heter_plot:
    heter_plot()

  if args.imagenet_plot:
    imagenet_plot()

  if args.framework_plot:
    framework_plot()

  if args.scalability_plot:
    scalability_plot()

  if args.summary:
    print_summary()

  if args.sync:
    sync_streamer(args.sync)
