#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time as timer
import time
import argparse
import configparser

from model.TinT import make_model
from lib.utils import get_adjacency_matrix, get_adjacency_matrix_2direction, compute_val_loss, predict_and_save_results, load_graphdata_normY_channel1
from tensorboardX import SummaryWriter

# read hyper-param settings
parser = argparse.ArgumentParser()


parser.add_argument("--config", default='TinT_configs.conf', type=str, help="configuration file path")

# parser.add_argument('--cuda', type=str, default='2')
# args.cuda = config['Training']['cudaID']

args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda


# USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device('cuda:0')
# print("CUDA:", USE_CUDA, DEVICE, flush=True)



config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config), flush=True)
config.read(args.config)
data_config = config['Data']
training_config = config['Training']



if int(training_config["multiGPU"]):
    os.environ['CUDA_VISIBLE_DEVICES'] = training_config["whichGPUs"]
    DEVICE = torch.device(f'cuda:0')
    training_config["DEVICE_ID"] = "0"
else:
    DEVICE = torch.device(f'cuda:{int(training_config["cudaID"])}')
    training_config["DEVICE_ID"] = training_config["cudaID"]




adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
dataset_name = data_config['dataset_name']
model_name = training_config['model_name']
learning_rate = float(training_config['learning_rate'])
start_epoch = int(training_config['start_epoch']) 
epochs = int(training_config['epochs'])
fine_tune_epochs = int(training_config['fine_tune_epochs'])
print('total training epoch, fine tune epoch:', epochs, ',' , fine_tune_epochs, flush=True)
batch_size = int(training_config['batch_size'])
print('batch_size:', batch_size, flush=True)
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
direction = int(training_config['direction'])
encoder_input_size = int(training_config['encoder_input_size'])
decoder_size = int(training_config['decoder_size'])
dropout = float(training_config['dropout'])
kernel_size = int(training_config['kernel_size'])

filename_npz = os.path.join(dataset_name + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '.npz'
num_layers = int(training_config['num_layers'])
d_model = int(training_config['d_model'])
nb_head = int(training_config['nb_head'])
# ScaledSAt = bool(int(training_config['ScaledSAt']))  # whether use spatial self attention
SE = bool(int(training_config['SE']))  # whether use spatial embedding
smooth_layer_num = int(training_config['smooth_layer_num'])
aware_temporal_context = bool(int(training_config['aware_temporal_context']))
TE = bool(int(training_config['TE']))
use_LayerNorm = True
residual_connection = True

# direction = 1 means: if i connected to j, adj[i,j]=1;
# direction = 2 means: if i connected to j, then adj[i,j]=adj[j,i]=1
if direction == 2:
    adj_mx, distance_mx = get_adjacency_matrix_2direction(adj_filename, num_of_vertices, id_filename)
if direction == 1:
    adj_mx, distance_mx = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename)
folder_dir = 'MAE_%s_h%dd%dw%d_layer%d_head%d_dm%d_channel%d_dir%d_drop%.2f_%.2e' % (model_name, num_of_hours, num_of_days, num_of_weeks, num_layers, nb_head, d_model, encoder_input_size, direction, dropout, learning_rate)

if aware_temporal_context:
    folder_dir = folder_dir+'Tcontext'
# if ScaledSAt:
#     folder_dir = folder_dir + 'ScaledSAt'
if SE:
    folder_dir = folder_dir + 'SE' + str(smooth_layer_num)
if TE:
    folder_dir = folder_dir + 'TE'

lt2 = time.strftime("%Y-%m-%d--%H_%M_%S", time.localtime())
folder_dir += str(lt2)
print('folder_dir:', folder_dir, flush=True)
params_path = os.path.join('experiments', dataset_name, folder_dir)

# all the input has been normalized into range [-1,1] by MaxMin normalization
train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_graphdata_normY_channel1(
    graph_signal_matrix_filename, num_of_hours,
    num_of_days, num_of_weeks, DEVICE, batch_size)


net = make_model(config, DEVICE, num_layers, encoder_input_size, decoder_size, d_model, adj_mx, nb_head, num_of_weeks,
                 num_of_days, num_of_hours, points_per_hour, num_for_predict, dropout=dropout, aware_temporal_context=aware_temporal_context, SE=SE, TE=TE, kernel_size=kernel_size, smooth_layer_num=smooth_layer_num, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)




if int(training_config["multiGPU"]):
    # torch.distributed.init_process_group(backend="nccl")
    # net = nn.parallel.DistributedDataParallel(net)#, device_ids=device_ids)
    device_ids = [int(i) for i in training_config["whichGPUs"].split(',')]
    net = nn.DataParallel(net, device_ids=device_ids)


def train_main():

    os.makedirs(params_path, exist_ok=1)
    print('create params directory %s' % (params_path), flush=True)


    criterion = nn.L1Loss().to(DEVICE)  # 定义损失函数
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # 定义优化器，传入所有网络参数
    sw = SummaryWriter(logdir=params_path, flush_secs=5)


    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf


    # train model
    if start_epoch > 0:


        mid_ckpt_path = '/home/winston/GCT/TinT/best_ckpt/best_ckpt_laRU.params'


        net.load_state_dict(torch.load(mid_ckpt_path, map_location=DEVICE))

        print('start epoch:', start_epoch, flush=True)

        print('load weight from: ', mid_ckpt_path, flush=True)

    start_time = timer()

    for epoch in range(start_epoch, epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        # apply model on the validation data set
        val_loss = compute_val_loss(net, val_loader, criterion, sw, epoch) #if epoch >0 else np.inf

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename, flush=True)
            predict_main(best_epoch, test_loader, test_target_tensor, _mean, _std, 'test')

        net.train()  # ensure dropout layers are in train mode

        train_start_time = timer()

        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs, decoder_inputs, labels = batch_data
            # encoder_inputs:
            # decoder_inputs:
            # labels:

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T) -> (B, N, T, 1)

            labels = labels.unsqueeze(-1)

            optimizer.zero_grad()

            outputs = net(encoder_inputs, decoder_inputs)

            loss = criterion(outputs, labels)

            if int(training_config["multiGPU"]):
                loss = loss.mean()

            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)

        # print('epoch: %s, train timer every whole data:%.2fs' % (epoch, timer() - train_start_time), flush=True)
        # print('epoch: %s, total timer:%.2fs' % (epoch, timer() - start_time), flush=True)
        print(f'train epoch: {epoch}, loss = {float(training_loss):.4f}')

    print('best epoch:', best_epoch, flush=True)

    print('apply the best val model on the test data set ...', flush=True)


    # fine tune the model
    optimizer = optim.Adam(net.parameters(), lr=learning_rate*0.1)
    print('fine tune the model ... ', flush=True)
    for epoch in range(epochs, epochs+fine_tune_epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        net.train()  # ensure dropout layers are in train mode

        train_start_time = timer()

        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs, decoder_inputs, labels = batch_data

            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1)
            predict_length = labels.shape[2]  # T

            optimizer.zero_grad()

            encoder_output = net.encode(encoder_inputs)

            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]
            decoder_input_list = [decoder_start_inputs]

            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]

            loss = criterion(predict_output, labels)

            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)

        print('epoch: %s, train timer every whole data:%.2fs' % (epoch, timer() - train_start_time), flush=True)
        print('epoch: %s, total timer:%.2fs' % (epoch, timer() - start_time), flush=True)

        # apply model on the validation data set
        val_loss = compute_val_loss(net, val_loader, criterion, sw, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename, flush=True)
            predict_main(best_epoch, test_loader, test_target_tensor, _mean, _std, 'test')
        print(f'fine tune epoch {epoch}')

    print('best epoch:', best_epoch, flush=True)

    print('apply the best val model on the test data set ...', flush=True)



def predict_main(epoch, data_loader, data_target_tensor, _mean, _std, type):
    '''
    在测试集上，测试指定epoch的效果
    :param epoch: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param _mean: (1, 1, 3, 1)
    :param _std: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

    print('load weight from:', params_filename, flush=True)

    net.load_state_dict(torch.load(params_filename))

    predict_and_save_results(net, data_loader, data_target_tensor, epoch, _mean, _std, params_path, type)


if __name__ == "__main__":

    train_main()















