import psutil
import subprocess


def get_num_gpus():
    command = 'nvidia-smi -L'
    output = subprocess.run(command, stdout=subprocess.PIPE, check=True,
                            shell=True).stdout.decode('utf-8').strip()
    return len(output.split('\n'))


def kill_pid_for_job(command):
    cmd_name = command.split(' ')[1]
    cmd_rank = command.split(' ')[-6]
    for proc in psutil.process_iter(['pid', 'cmdline', 'status']):
        proc_cmdline = proc.info['cmdline']
        if cmd_name in proc_cmdline and cmd_rank in proc_cmdline and "/bin/sh" not in proc_cmdline:
            try:
                proc.terminate()
            except psutil.NoSuchProcess:
                print(f"Process with name {proc} not found.")
            except psutil.AccessDenied:
                print(f"Access denied to terminate process {proc} with PID: {proc.info['pid']}")