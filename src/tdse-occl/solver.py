import time
from utils import *
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.nn.functional as F


class Solver(object):
    def __init__(self, train_data, validation_data, test_data, model, optimizer, args):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.args = args
        self.amp = amp
        self.v_loss = nn.MSELoss()

        self.print = False
        if (self.args.distributed and self.args.local_rank ==0) or not self.args.distributed:
            self.print = True
            if self.args.use_tensorboard:
                self.writer = SummaryWriter('logs/%s/tensorboard/' % args.log_name)

        self.model, self.optimizer = self.amp.initialize(model, optimizer,
                                                        opt_level=args.opt_level,
                                                        patch_torch_functions=args.patch_torch_functions)

        if self.args.distributed:
            self.model = DDP(self.model)

        self._reset()

    def _reset(self):
        self.halving = False
        if self.args.continue_from:
            checkpoint = torch.load('logs/%s/model_dict_last.pt' % self.args.continue_from, map_location='cpu')

            
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.amp.load_state_dict(checkpoint['amp'])

            self.start_epoch=checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.val_no_impv = checkpoint['val_no_impv']

            if self.print: print("Resume training from epoch: {}".format(self.start_epoch))
            
        else:
            self.best_val_loss = float("inf")
            self.val_no_impv = 0
            self.start_epoch=1
            if self.print: print('Start new training')

    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs+1):
            if self.args.distributed: self.args.train_sampler.set_epoch(epoch)
#             Train
            self.model.train()
            start = time.time()
            tr_loss, tr_v_loss = self._run_one_epoch(data_loader = self.train_data, state='train')
            reduced_tr_loss = self._reduce_tensor(tr_loss)
            reduced_tr_v_loss = self._reduce_tensor(tr_v_loss)

            if self.print: print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Train Loss {2:.3f}'.format(
                        epoch, time.time() - start, reduced_tr_loss))

            # Validation
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                val_loss, val_v_loss = self._run_one_epoch(data_loader = self.validation_data, state='val')
                reduced_val_loss = self._reduce_tensor(val_loss)
                reduced_val_v_loss = self._reduce_tensor(val_v_loss)

            if self.print: print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Valid Loss {2:.3f}'.format(
                          epoch, time.time() - start, reduced_val_loss))

            # test
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                test_loss, test_v_loss = self._run_one_epoch(data_loader = self.test_data, state='test')
                reduced_test_loss = self._reduce_tensor(test_loss)
                reduced_test_v_loss = self._reduce_tensor(test_v_loss)

            if self.print: print('Test Summary | End of Epoch {0} | Time {1:.2f}s | '
                      'Test Loss {2:.3f}'.format(
                          epoch, time.time() - start, reduced_test_loss))


            # Check whether to adjust learning rate and early stop
            find_best_model = False
            if reduced_val_loss >= self.best_val_loss:
                self.val_no_impv += 1
                if self.val_no_impv >= 10:
                    if self.print: print("No imporvement for 10 epochs, early stopping.")
                    break
            else:
                self.val_no_impv = 0
                self.best_val_loss = reduced_val_loss
                find_best_model=True

            if self.val_no_impv == 6:
                self.halving = True

            # Halfing the learning rate
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] /2
                self.optimizer.load_state_dict(optim_state)
                if self.print: print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False

            if self.print:
                # Tensorboard logging
                if self.args.use_tensorboard:
                    self.writer.add_scalar('Train_loss', reduced_tr_loss, epoch)
                    self.writer.add_scalar('Validation_loss', reduced_val_loss, epoch)
                    self.writer.add_scalar('Test_loss', reduced_test_loss, epoch)

                    self.writer.add_scalar('Train_v_loss', reduced_tr_v_loss, epoch)
                    self.writer.add_scalar('Validation_v_loss', reduced_val_v_loss, epoch)
                    self.writer.add_scalar('Test_v_loss', reduced_test_v_loss, epoch)

                # Save model
                checkpoint = {'model': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'amp': self.amp.state_dict(),
                                'epoch': epoch+1,
                                'best_val_loss': self.best_val_loss,
                                'val_no_impv': self.val_no_impv}
                torch.save(checkpoint, "logs/"+ self.args.log_name+"/model_dict_last.pt")
                if find_best_model:
                    torch.save(checkpoint, "logs/"+ self.args.log_name+"/model_dict_best.pt")
                    print("Fund new best model, dict saved")


    def _run_one_epoch(self, data_loader, state):
        total_loss = 0
        total_v_loss = 0
        self.accu_count = 0
        self.optimizer.zero_grad()
        for i, (a_mix, a_tgt, v_tgt, v_tgt_gt) in enumerate(data_loader):
            a_mix = a_mix.cuda().squeeze(0).float()
            a_tgt = a_tgt.cuda().squeeze(0).float()
            v_tgt = v_tgt.cuda().squeeze(0).float()
            v_tgt_gt = v_tgt_gt.cuda().squeeze(0).float()

            est_a_tgt, v_reconstruct = self.model(a_mix, v_tgt)
            max_snr = cal_SISNR(a_tgt, est_a_tgt)

            v_loss = 0
            v_tgt_gt = v_tgt_gt.transpose(1,2)
            for k in range(len(v_reconstruct)):
                v_out = v_reconstruct[k]
                v_out = F.pad(v_out,(0,v_tgt_gt.size(2)-v_out.size(2)))
                v_loss += self.v_loss(v_out, v_tgt_gt)

            if state =='train':
                loss = 0 - torch.mean(max_snr) + self.args.gamma*v_loss
  
                if state == 'train':
                    self.accu_count += 1
                    if self.args.accu_grad:
                        loss = loss/(self.args.effec_batch_size / self.args.batch_size)
                        with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.amp.master_params(self.optimizer),
                                                       self.args.max_norm)
                        if self.accu_count == (self.args.effec_batch_size / self.args.batch_size):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            self.accu_count = 0
                    else:
                        with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.amp.master_params(self.optimizer),
                                                       self.args.max_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

            elif state =='val':
                loss = 0 - torch.mean(max_snr)

            else:
                loss = 0 - torch.mean(max_snr[::self.args.C])


            total_loss += loss.data
            total_v_loss += v_loss.data

        return total_loss / (i+1), total_v_loss / (i+1)

    def _reduce_tensor(self, tensor):
        if not self.args.distributed: return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt

