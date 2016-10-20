# eval.py - Evaluate the performance of streamer
from termcolor import colored
from subprocess import call
from argparse import ArgumentParser


LOCAL_DIR = "/Users/xianran/Code/TX1/streamer/build"
LOCAL_HOST_DIR = "localhost:/Users/xianran/Code/TX1/streamer/build"
NUC_HOST_DIR = "/Users/xianran/Code/TX1/streamer/build"
TEGRA_HOST_DIR = "tegra-1:/home/ubuntu/Code/tx1dnn/build"

DEFAULT_NETWORK = "GoogleNet"


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
