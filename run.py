import os
import numpy as np

if not os.path.exists('log'):
    os.mkdir('log')
dataset = 'duke'
for i in np.arange(1, 10):
    print('i = %.3f' % i)
    log_name = 'log/' + 'log_' + str(i)
    print('log name = %s' % log_name)

    os.system('python self_supervise_augment.py  --data_dir duke  --mode ' + str(i) + ' >> ' + log_name)

    cmd = 'python train.py --save_model_name ' + str(
        i) + ' --data_dir ' + dataset + ' --use_dense --gpu_ids 0  --net_loss_model ' + str(
        i) + ' --name sggnn --train_all --batchsize  48  --erasing_p 0.5' + ' >> ' + log_name
    print('cmd = %s' % cmd)
    os.system(cmd)

    os.system(
        'python test_sggnn.py  --test_dir ' + dataset + ' --use_dense --gpu_ids 0 --name sggnn --which_epoch best_siamese ' + ' >>  ' + log_name)
    os.system('python evaluate_gpu.py' + ' >> ' + log_name)
    os.system('python evaluate_rerank.py' + ' >> ' + log_name)

    os.system(
        'python test_sggnn.py  --test_dir ' + dataset + ' --use_dense --gpu_ids 0 --name sggnn --which_epoch last_siamese ' + ' >>  ' + log_name)
    os.system('python evaluate_gpu.py' + ' >> ' + log_name)
    os.system('python evaluate_rerank.py' + ' >> ' + log_name)
