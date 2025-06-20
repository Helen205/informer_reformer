import argparse
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from exp.exp_informer import Exp_Informer

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

parser.add_argument('--data', type=str, default='CSI300', help='data')
parser.add_argument('--root_path', type=str, default='./data', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='CSI300.csv', help='data file')    
parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='Close', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='d', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=60, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=30, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=20, help='prediction sequence length')



parser.add_argument('--scale', type=bool, default=True, help='Whether to scale the data (True/False)')


parser.add_argument('--layer_normalization', type=bool, default=True, help='Whether to scale the data (True/False)')
parser.add_argument('--timeenc', type=int, default=1, help='Time encoding type (0: no encoding, 1: positional encoding)')

parser.add_argument('--enc_in', type=int, default=14, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=14, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=3, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='relu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.003, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='huber',help='loss function')
parser.add_argument('--optimizer', type=str, default='adamw',help='optimizer to use (adam, sgd, rmsprop, adamw, adagrad, adadelta)')
#parser.add_argument('--momentum', type=float, default=0.9,help='momentum for SGD optimizer')
#parser.add_argument('--weight_decay', type=float, default=0.12,help='weight decay for AdamW optimizer')
#parser.add_argument('--alpha', type=float, default=0.99,help='alpha parameter for RMSprop optimizer')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'CSI300':{'data':'CSI300.csv','T':'Close','M':[14,14,14],'S':[1,1,1],'MS':[14,14,1]} 
}


print('Args in experiment:')
print(args)

Exp = Exp_Informer
writer = SummaryWriter(log_dir='./logs')
for ii in range(args.itr):
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des, ii)
    
    
    
    exp = Exp(args)
    train_loss, val_loss = exp.train(setting)  
    test_loss = exp.test(setting)
    
    writer.add_scalar("Train Loss", train_loss, ii)
    writer.add_scalar("Validation Loss", val_loss, ii)
    writer.add_scalar("Test Loss", test_loss[0], ii)



    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
    
    save_path = os.path.join(args.checkpoints, f"{setting}_model.pth")
    torch.save(exp.model.state_dict(), save_path)
    print(f"Model saved to {save_path}")



    torch.cuda.empty_cache()

writer.close()




