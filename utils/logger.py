import time
import os
import logging
import fcntl
import csv
import pandas as pd

def get_logger(path, suffix):
    cur_time = time.strftime('%Y-%m-%d-%H.%M.%S',time.localtime(time.time()))
    logger = logging.getLogger(__name__+cur_time)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(os.path.join(path, f"{suffix}_{cur_time}.log"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


class ResultRecorder(object):
    def __init__(self, path, total_cv=3):
        self.path = path
        self.total_cv = total_cv
    
    def is_full(self):

        df = pd.read_csv(self.path)
        if len(df) < self.total_cv:
            return False
        return True
    
    def calc_mean(self):
        
        df = pd.read_csv(self.path)
        mean_res= {}
        for item in df.columns:
            mean_res[item] = df[item].mean()
        
        with open(self.path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([f"{value:.4f}" for value in mean_res.values()])
        

    def write_result_to_tsv(self, results, cvNo):
        if os.path.exists(self.path):
            with open(self.path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([f"{value:.4f}" for value in results.values()])
        else:
            with open(self.path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(results.keys())
                writer.writerow([f"{value:.4f}" for value in results.values()])
        # if self.is_full():
        #     self.calc_mean()
            

# class ResultRecorder(object):
#     def __init__(self, path, total_cv=3):
#         self.path = path
#         self.total_cv = total_cv
#         if not os.path.exists(self.path):
#             f = open(self.path, 'w')
#             # f.write('acc\tuar\tf1\n')
#             f.write('Acc_2\tF1_score\tAcc_7\tMAE\tCorr\n')
#             f.close()
    
#     def is_full(self, content):
#         if len(content) < self.total_cv+1:
#             return False
        
#         for line in content:
#             if not len(line.split('\t')) == 3:
#                 return False
#         return True
    
#     def calc_mean(self, content):
        
#         acc2_metric = [float(line.split('\t')[0]) for line in content[1:]]
#         f1_metric = [float(line.split('\t')[1]) for line in content[1:]]
#         acc7_metric = [float(line.split('\t')[2]) for line in content[1:]]
#         mae_metric = [float(line.split('\t')[3]) for line in content[1:]]
#         corr_metric = [float(line.split('\t')[4]) for line in content[1:]]


#         mean_acc2_metric = sum(acc2_metric) / len(acc2_metric)
#         mean_f1_metric = sum(f1_metric) / len(f1_metric)
#         mean_acc7_metric = sum(acc7_metric) / len(acc7_metric)
#         mean_mae_metric = sum(mae_metric) / len(mae_metric)
#         mean_corr_metric = sum(corr_metric) / len(corr_metric)
#         return mean_acc2_metric, mean_f1_metric, mean_acc7_metric,mean_mae_metric,mean_corr_metric

#     def write_result_to_tsv(self, results, cvNo):
#         # 使用fcntl对文件加锁,避免多个不同进程同时操作同一个文件
#         f_in = open(self.path)
#         fcntl.flock(f_in.fileno(), fcntl.LOCK_EX) # 加锁
#         content = f_in.readlines()
#         if len(content) < self.total_cv+1:
#             content += ['\n'] * (self.total_cv-len(content)+1)
#         keys = [item for item in results.keys()]
#         content[cvNo] = '{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(results[keys[0]], results[keys[1]], results[keys[2]], results[keys[3]], results[keys[4]])
#         if self.is_full(content):
#             mean_acc2_metric, mean_f1_metric, mean_acc7_metric,mean_mae_metric,mean_corr_metric = self.calc_mean(content)
#             content.append('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(mean_acc2_metric, mean_f1_metric, mean_acc7_metric,mean_mae_metric,mean_corr_metric))

#         f_out = open(self.path, 'w')
#         f_out.writelines(content)
#         f_out.close()
#         f_in.close()                              # 释放锁


class LossRecorder(object):
    def __init__(self, path, total_cv=10, total_epoch=40):
        self.path = path
        self.total_epoch = total_epoch
        self.total_cv = total_cv
        if not os.path.exists(self.path):
            f = open(self.path, 'w')
            f.close()

    def is_full(self, content):
        if len(content) < self.total_cv + 1:
            return False

        for line in content:
            if not len(line.split('\t')) == 3:
                return False
        return True

    def calc_mean(self, content):
        loss_list = [[] * self.total_cv] * self.total_epoch
        mean_list = [[] * self.total_cv] * self.total_epoch
        for i in range(0, self.total_epoch):
            loss_list[i] = [float(line.split('\t')[i]) for line in content[1:]]
        for i in range(0, self.total_epoch):
            mean_list[i] = sum(loss_list[i]) / len(loss_list[i])
        return mean_list

    def write_result_to_tsv(self, results, cvNo):
        # 使用fcntl对文件加锁,避免多个不同进程同时操作同一个文件
        f_in = open(self.path)
        fcntl.flock(f_in.fileno(), fcntl.LOCK_EX)  # 加锁
        content = f_in.readlines()
        if len(content) < self.total_cv + 1:
            content += ['\n'] * (self.total_cv - len(content) + 1)
        string = ''
        for i in results:
            string += str(i.numpy())[:8]
            string += '\t'
        content[cvNo] = string + '\n'

        f_out = open(self.path, 'w')
        f_out.writelines(content)
        f_out.close()
        f_in.close()  # 释放锁

    def read_result_from_tsv(self,):
        f_out = open(self.path)
        fcntl.flock(f_out.fileno(), fcntl.LOCK_EX)
        content = f_out.readlines()
        loss_list = [[] * self.total_cv] * self.total_epoch
        for i in range(0, self.total_epoch):
            loss_list[i] = [float(line.split('\t')[i]) for line in content[1:]]
        mean = self.calc_mean(content)
        return mean
