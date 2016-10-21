# eval.py - Evaluate the performance of streamer
from termcolor import colored
import subprocess
from threading import Thread
from subprocess import call
from subprocess import Popen
from argparse import ArgumentParser
import numpy as np
import time


LOCAL_DIR = "/Users/xianran/Code/TX1/streamer/build"
LOCAL_HOST_DIR = "localhost:/Users/xianran/Code/TX1/streamer/build"
NUC_HOST_DIR = "/Users/xianran/Code/TX1/streamer/build"
TEGRA_HOST_DIR = "tegra-1:/home/ubuntu/Code/tx1dnn/build"

# Default values
DEFAULT_NETWORK = "GoogleNet"
DEFAULT_CAMERA = "GST_TEST"
DEFAULT_ITERATION = 100


class MetricCollector(object):

  """Collector for cpu, memory and gpu stats"""

  def __init__(self, host, on_tegra=False, on_nuc=False, on_mac=False):
    self.host = host
    self.on_tegra = on_tegra
    self.on_nuc = on_nuc
    self.on_mac = on_mac
    self.readout_thread = None

    self.mem = 0.0
    self.cpu = 0.0
    self.gpu = 0.0

    self.stopped = True

  def get_pid_(self):
    return self.execute_command_("pidof benchmark")[0]

  def execute_command_(self, cmd):
    ssh_command = "ssh %s 'zsh -l -c \"%s\"'" % (self.host, cmd)
    process = Popen(
        ssh_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return process.communicate()

  def start(self):
    self.readout_thread = Thread(target=self.collect_stats_)
    self.stopped = False
    self.readout_thread.start()

  def process_tegra_(self, line):
    self.results = []

    sp = line.split(" ")
    mem_str = sp[1].split("/")[0]
    cpu_str = sp[5][1:-6]
    gpu_str = sp[-4]

    mem = float(mem_str)
    gpu = float(gpu_str.split("@")[0][:-1])
    cpu = 0.0

    for c in cpu_str.split(","):
      cpu += float(c[:-1])

    return mem, cpu, gpu

  def collect_stats_(self):
    while not self.stopped:
      pid = self.get_pid_()
      if pid == '' or pid == None:
        continue
      # Get stats by executing ps aux on pid
      collect_stats_cmd = "ps aux -q " + pid
      output, _ = self.execute_command_(collect_stats_cmd)

      # Remove empty strings
      if output == '':
        continue
      output = output.split("\n")[1]
      if output == '':
        break
      output = filter(bool, output.split(" "))
      if output[-1] == '<defunct>':
        break

      self.cpu = float(output[2])
      if output[5] != '0':
        self.mem = float(output[5])

  def stop(self):
    self.stopped = True
    self.readout_thread.join()
    self.readout_thread = None

  def get_cpu_usage(self):
    return self.cpu

  def get_gpu_usage(self):
    return self.gpu

  def get_mem_usage(self):
    return self.mem


def print_error(msg):
  """Print message in alert color"""
  print colored(msg, 'red')


def sync_codebase(host_dir):
  ssh_command = "rsync -avh --delete --exclude=.git/* --exclude=build/* --exclude=.idea/* --exclude=config/* %s/.. %s/.. > /dev/null" % (
      LOCAL_DIR, host_dir)
  if call(ssh_command, shell=True) != 0:
    print_error("Can't sync codebase with: " + host_dir)
    exit(-1)


def remote_execute(host_dir, *cmd):
  host, streamer_dir = host_dir.split(":")
  ssh_command = "ssh %s 'zsh -l -c \"(cd %s; %s)\"'" % (
      host, streamer_dir, ' '.join(cmd))
  p = Popen(
      ssh_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  output = p.communicate()

  if p.returncode != 0:
    print_error("Error executing: %s on %s" % (host_dir, cmd))
    exit(-1)

  return output


def build_streamer(host_dir):
  remote_execute(host_dir, "make -j")


def run_experiment(host_dir, network=DEFAULT_NETWORK, camera=DEFAULT_CAMERA, on_tegra=False, on_nuc=False):
  def grab_fps(result, pattern):
    for line in result.split("\n"):
      if pattern in line and "fps" in line:
        return float(line.split(" ")[-1])
    return 0.0

  sync_codebase(host_dir)
  build_streamer(host_dir)

  remote_execute(host_dir, "sleep 1")

  host, _ = host_dir.split(":")
  collector = MetricCollector(host, on_tegra=on_tegra)
  collector.start()

  output, _ = remote_execute(host_dir,
                             "GLOG_minloglevel=3 apps/benchmark",
                             '--experiment', 'classification',
                             '--camera', camera,
                             '--net', network,
                             '--verbose', 'true',
                             '--iter', str(DEFAULT_ITERATION))
  collector.stop()
  print 'cpu', collector.get_cpu_usage()
  print 'mem', collector.get_mem_usage()
  print 'fps', grab_fps(output, "classifier")


def heter_eval():
  run_experiment(TEGRA_HOST_DIR, on_tegra=True)
  # run_experiment(LOCAL_HOST_DIR)


def heter_plot():
  pass

if __name__ == "__main__":
  parser = ArgumentParser("Evaluate performance of streamer")
  parser.add_argument(
      "--heter_eval", help="Run heterogeneous experiment", action='store_true')
  parser.add_argument(
      "--heter_plot", help="Plot heterogeneous experiment", action='store_true')

  args = parser.parse_args()

  if args.heter_eval:
    heter_eval()

  if args.heter_plot:
    heter_plot()
