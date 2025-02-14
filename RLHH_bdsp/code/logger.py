import sys
import os
import time


# 控制台输出记录到文件
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def save_log_file(log_file_name=None):
    log_path = '../Logs/'
    os.makedirs(log_path, exist_ok=True)
    if log_file_name is None:
        log_file_name = 'log-'
    log_file = log_path + log_file_name + '-' + time.strftime("%H.%M-%m.%d", time.localtime()) + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file)

if __name__ == '__main__':
    save_log_file()

    print(5555)
    print(2/0)
