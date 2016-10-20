# eval.py - Evaluate the performance of streamer
from termcolor import colored
import subprocess
from subprocess import call
from subprocess import Popen
from argparse import ArgumentParser
import numpy as np
import time


LOCAL_DIR = "/Users/xianran/Code/TX1/streamer/build"
LOCAL_HOST_DIR = "localhost:/Users/xianran/Code/TX1/streamer/build"
NUC_HOST_DIR = "/Users/xianran/Code/TX1/streamer/build"
TEGRA_HOST_DIR = "tegra-1:/home/ubuntu/Code/tx1dnn/build"

DEFAULT_NETWORK = "GoogleNet"


class MetricCollector(object):

  def __init__(self, host, on_tegra=False, on_nuc=False):
    self.host = host
    self.on_tegra = on_tegra
    self.on_nuc = on_nuc
    self.sampler_process = None

    self.mems = []
    self.cpus = []
    self.gpus = []

  def execute_command_(self, cmd, wait=True):
    ssh_command = "ssh %s 'zsh -l -c \"%s\"'" % (self.host, cmd)
    if wait:
      self.sampler_process = Popen(
          ssh_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
      call(ssh_command, shell=True)

  def start(self):
    if self.on_tegra:
      ssh_command = "sudo pkill -9 -f *tegrastats*"
      self.execute_command_(ssh_command, wait=True)
      ssh_command = "sudo ~/tegrastats"
      self.execute_command_(ssh_command)
    elif self.on_nuc:
      pass
    else:
      pass

  def process_tegra_(self, line):
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

  def process_(self):
    for line in self.results:
      if self.on_tegra:
        mem, cpu, gpu = self.process_tegra_(line)
        self.mems.append(mem)
        self.cpus.append(cpu)
        self.gpus.append(gpu)
      elif self.on_nuc:
        pass
      else:
        pass

  def stop(self):
    self.results = []
    self.sampler_process.terminate()
    while True:
      line = self.sampler_process.stdout.readline()
      if line == None or line == '':
        break
      self.results.append(line.strip())

    self.process_()

  def get_cpu_usage(self):
    return np.mean(self.cpus)

  def get_gpu_usage(self):
    return np.mean(self.gpus)

  def get_mem_usage(self):
    return np.mean(self.mems)


def print_error(msg):
  print colored(msg, 'red')


def sync_codebase(host_dir):
  ssh_command = "rsync -avh --delete --exclude=.git/* --exclude=build/* --exclude=.idea/* --exclude=config/* %s/.. %s/.." % (
      LOCAL_DIR, host_dir)
  if call(ssh_command, shell=True) != 0:
    print_error("Can't sync codebase with: " + host_dir)
    exit(-1)


def remote_execute(host_dir, cmd):
  host, streamer_dir = host_dir.split(":")
  ssh_command = "ssh %s 'zsh -l -c \"(cd %s; %s)\"'" % (
      host, streamer_dir, cmd)
  print ssh_command
  if call(ssh_command, shell=True) != 0:
    print_error("Error executing: %s on %s" % (host_dir, cmd))
    exit(-1)


def build_streamer(host_dir):
  remote_execute(host_dir, "make -j")


def run_experiment(host_dir, network=DEFAULT_NETWORK):
  sync_codebase(host_dir)
  build_streamer(host_dir)
  remote_execute(host_dir, "apps/benchmark")


def heter_eval():
  run_experiment(LOCAL_HOST_DIR)


def heter_plot():
  pass

if __name__ == "__main__":
  collector = MetricCollector(TEGRA_HOST_DIR.split(":")[0], on_tegra=True)
  collector.start()
  time.sleep(4)
  collector.stop()

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
