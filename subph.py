import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import csv
from scipy.optimize import minimize


def load_data(file_path):
    # 读取CSV文件
    # read the CSV file
    df = pd.read_csv(file_path)
    # 获取时间t, x位置和y位置，并转换为Tensor，同时确保数据类型为torch.float32
    # Retrieve time t, x, y, and convert them to Tensors, ensuring that the data type is torch.float32
    t_obs = df['Time'].values
    x_obs = df['X'].values
    y_obs = df['Y'].values
    t_start = t_obs[0] # 训练集数据和测试集数据的时间都是从0开始的
    t_end = t_obs[-1]
    return t_start, t_end, t_obs, x_obs, y_obs


# 定义重力加速度
# define g
g = 9.8

nodes = 32


# 定义神经网络，输入为时间t，输出为轨迹信息x, y坐标以及阻力参数k。
# 你可以采用别的定义方法，这里只是给出了一种可行的方式
# Define the neural network, with input as time t , and output as trajectory information including  x  and  y  coordinates, as well as the drag parameter k .
# You can use a different method; this is just one possible approach.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(1, nodes)

        self.output_layer_x = nn.Linear(nodes, 1)
        self.output_layer_y = nn.Linear(nodes, 1)
        self.output_layer_k = nn.Linear(nodes, 1)

    def forward(self, t):
        outputs = torch.tanh(self.hidden_layer1(t))

        x = self.output_layer_x(outputs)
        y = self.output_layer_y(outputs)
        k = self.output_layer_k(outputs)
        return x, y, k


# 定义损失函数，包括物理损失与数据损失
# Define the loss function, including physical loss and data loss.

def loss(parameter,data): #2,3,4
    x_train, y_train, t_train = data[0], data[1], data[2]
    kb, vxb, vyb = parameter[0], parameter[1], parameter[2]
    generated = generate(vxb,vyb,kb,t_train,300,x_train[0],y_train[0],g,1)
    generate_x = generated[1]
    generate_y = generated[2]
    total_loss = sum((np.array(generate_x) - np.array(x_train))**2) + sum((np.array(generate_y) - np.array(y_train))**2)
    return total_loss


def export_parameter(net, submission_path):
    # 阻力系数，由你通过训练得出
    # Drag coefficient, derived from your training.
    k_pred = net[2]
    # 最大高度，由你通过训练得出
    # Maximum height, derived from your training.
    height_max = net[1]
    # 最远射程，由你通过训练得出
    # Range, derived from your training.
    range_max = net[0]
    with open(submission_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['k', 'height_max', 'range_max'])
        writer.writerow([k_pred, height_max, range_max])


def calculate_parameters(data_path, submission_path):#t_start, t_end, t_obs, x_obs, y_obs
    t_start, t_end, t_obs, x_obs, y_obs = load_data(data_path)
    #gen = generate(100, 100, 0, t_obs, 10000, x_obs[0], y_obs[0], 1, 1)
    #print("t",gen[0])
    resul = minimize(loss,[1,1,1],args=([x_obs, y_obs, t_obs]),method='Nelder-Mead')
    result_k,result_vx,result_vy = resul['x'][0],resul['x'][1],resul['x'][2] # kb, vxb, vyb = parameter[0], parameter[1], parameter[2]
    #print("k,vx,vy",result_k,result_vx,result_vy)

    #result_g = generate(result_vx,result_vy,result_k,t_obs,300,x_obs[0],y_obs[0],g,1)
    #for i in range(len(result_g[0])):
        #print(result_g[0][i],result_g[1][i],result_g[2][i],"===",t_obs[i],x_obs[i],y_obs[i])
    mx = max_x(result_vx, result_vy, result_k, t_obs[0], 0.000001, x_obs[0], y_obs[0], g, 1)
    my = max_y(result_vx, result_vy, result_k, t_obs[0], 0.000001, x_obs[0], y_obs[0], g, 1)
    #print(mx)
    #print(my)
    export_parameter([mx,my,result_k], submission_path)

def generate(vx0, vy0, k, t, num, x0, y0, g, m):
    curt = t[0]
    n = 1
    vx = vx0
    vy = vy0
    listx = [x0]
    listy = [y0]
    listt = [t[0]]
    x = x0
    y = y0

    for i in range(1,len(t)):
        deltat = (t[n] - t[n-1]) / num #num n
        for i in range (num):
            curt = curt + deltat
            vx = vx - k * vx * np.sqrt(vx**2 + vy**2) * deltat / m
            vy = vy - g * deltat - k * vy * np.sqrt(vx**2 + vy**2) * deltat / m
            x = vx*deltat + x
            y = vy*deltat + y

        listt.append(curt)
        listx.append(x)
        listy.append(y)
        n += 1
    #print(len(listt))
    return listt, listx, listy

#---------提交范例-------#
#---------Submission example-------#
import zipfile
import os
#导入训练集数据
# import training data
def max_x(vx0, vy0, k, t0, deltat, x0, y0, g, m):
    vx = vx0
    vy = vy0
    x = x0
    y = y0

    curt = t0
    while y >= 0:
            curt = curt + deltat
            vx = vx - k * vx * np.sqrt(vx**2 + vy**2) * deltat / m
            vy = vy - g * deltat - k * vy * np.sqrt(vx**2 + vy**2) * deltat / m
            x = vx*deltat + x
            y = vy*deltat + y
    return x
def max_y(vx0, vy0, k, t0, deltat, x0, y0, g, m):
    vx = vx0
    vy = vy0
    x = x0
    y = y0

    curt = t0
    while vy >= 0:
        curt = curt + deltat
        vx = vx - k * vx * np.sqrt(vx ** 2 + vy ** 2) * deltat / m
        vy = vy - g * deltat - k * vy * np.sqrt(vx ** 2 + vy ** 2) * deltat / m
        x = vx * deltat + x
        y = vy * deltat + y
    #print(vy)
    return y

#---------提交范例-------#
#---------Submission example-------#
import zipfile
import os
#导入训练集数据
# import training data

#TRAIN_PATH = "/bohr/train-g6oc/v1/"
TRAIN_PATH = "D:/AI4S_Teen_Cup_2025/dataset/Physics/"
calculate_parameters(data_path = TRAIN_PATH + "projectile_train.csv", submission_path = "submission_train.csv") #求解训练集方程

# 导入测试集数据
#“DATA_PATH”是测试集加密后的环境变量，按照如下方式可以在提交后，系统评分时访问测试集，但是选手无法直接下载
# import test data
#"DATA_PATH" is an encrypted environment variable for the test set. It can be accessed by the system for scoring after submission, but contestants cannot directly download it.

if os.environ.get('DATA_PATH'):
    DATA_PATH = os.environ.get("DATA_PATH") + "/"
else:
    #Baseline运行时，因为无法读取测试集，所以会有此条报错，属于正常现象
    #During the execution of the Baseline, since it is unable to read the test set, this error message will appear, which is a normal phenomenon
    print("During the execution of the Baseline, since it is unable to read the test set, this error message will appear, which is a normal phenomenon.")
DATA_PATH = "D:/AI4S_Teen_Cup_2025/dataset/Physics/"
calculate_parameters(data_path = DATA_PATH + "projectile_testA.csv", submission_path = "submission_testA.csv") #求解Public测试集方程
calculate_parameters(data_path = DATA_PATH + "projectile_testB.csv", submission_path = "submission_testB.csv") #求解Private测试集方程


# 定义要打包的文件和压缩文件名
files_to_zip = ['submission_testA.csv', 'submission_testB.csv']
zip_filename = 'submission.zip'

# 创建一个 zip 文件
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for file in files_to_zip:
        # 将文件添加到 zip 文件中
        zipf.write(file, os.path.basename(file))

print(f'{zip_filename} 创建成功!')