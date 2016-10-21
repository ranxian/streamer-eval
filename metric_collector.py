import subprocess
from threading import Thread
from subprocess import call
from subprocess import Popen


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
    self.stopped = False
    self.readout_thread = Thread(target=self.collect_stats_)
    self.readout_thread.daemon = True
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
