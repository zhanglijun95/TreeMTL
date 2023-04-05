import numpy as np
from sys import exit
import torch
import torch.nn as nn
from .trainer import Trainer

class DomainNetTrainer(Trainer):
    def __init__(self, model, tasks, train_dataloader, val_dataloader, batch_size=32, 
                 lr=0.001, decay_lr_freq=2000, decay_lr_rate=0.3,
                 print_iters=100, val_iters=200):
        
        self.model = model
        self.startIter = 0
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_lr_freq, gamma=decay_lr_rate)
        
        self.tasks = tasks
        self.batch_size = batch_size
        
        self.train_dataloader = train_dataloader
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = val_dataloader
        self.cross_entropy = nn.CrossEntropyLoss()
        self.loss_list = {}
        self.set_train_loss()
        
        self.print_iters = print_iters
        self.val_iters = val_iters
        self.best_records = None
        
    def train(self, iters=20000, loss_lambda=None, savePath=None, reload=None, writerPath=None):
        if writerPath != None:
            writer = SummaryWriter(log_dir=writerPath)
        else:
            writer = None
        
        self.model.train()
        if reload is not None and reload != 'false' and savePath is not None:
            self.load_model(savePath, reload)
        if loss_lambda is None:
            loss_lambda = {task: 1 for task in self.tasks}

        for i in range(self.startIter, iters):
            self.train_step(loss_lambda)
            if (i+1) % self.print_iters == 0:
                self.print_train_loss(i, writer)
                self.set_train_loss()
            if (i+1) % self.val_iters == 0:
                val_results = self.validate(i, writer)
                self.save_best(val_results, i, savePath)
                
        # Reset loss list
        self.set_train_loss()
        return
    
    def train_step(self, loss_lambda):
        self.model.train()
        try:
            batch = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_dataloader)
            batch = next(self.train_iter)
        
        data = self.organize_batch(batch)
        self.optimizer.zero_grad()
        loss = 0
        for task in self.tasks:
            x, y = data['%s_img' % task].cuda(), data['%s_gt' % task].cuda()
            output = self.model(x, task)
            tloss = self.cross_entropy(output, y)
            self.loss_list[task].append(tloss.item())
            loss += loss_lambda[task] * tloss
        self.loss_list['total'].append(loss.item())
        
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return
    
    def validate(self, it=0, writer=None):
        self.model.eval()
        loss_list = {task:[] for task in self.tasks}
        acc_list = {task:[] for task in self.tasks}
        total_img = 0
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_dataloader):
                data = self.organize_batch(batch)
                for task in self.tasks:
                    x, y = data['%s_img' % task].cuda(), data['%s_gt' % task].cuda()
                    y = y.view(-1)
                    output = self.model(x, task)
                    # loss
                    tloss = self.cross_entropy(output, y)
                    loss_list[task].append(tloss.item())
                    # acc.
                    prediction = torch.argmax(output, dim=1).view(-1)
                    acc = (y == prediction).float().sum()
                    acc_list[task].append(acc.cpu().numpy())
                total_img += len(data['%s_img' % self.tasks[0]])
                # total_img += self.batch_size
        val_return = {}
        for task in self.tasks:
            avg_loss = np.mean(loss_list[task])
            avg_acc = np.sum(acc_list[task])/total_img
            val_return[task] = avg_acc
            if writer != None:
                writer.add_scalar('Loss/val/' + task, avg_loss, it)
                writer.add_scalar('Acc/val/' + task, avg_acc, it)
            print('[Iter {} Task {}] Val Loss: {:.4f} Val Acc: {:.4f}'.format((it+1), task[:4], avg_loss, avg_acc), flush=True)
        print('======================================================================', flush=True)
        return val_return
    
    ################## helper functions ################
    def organize_batch(self, batch):
        assert batch['img'].shape[0] == self.batch_size * len(self.tasks)
        assert len(batch['img_idx']) == self.batch_size * len(self.tasks)
        new_batch = {}
        for d_idx, task in enumerate(self.tasks):
            new_batch['%s_img' % task] = batch['img'][d_idx * self.batch_size: (d_idx + 1) * self.batch_size]
            new_batch['%s_gt' % task] = batch['img_idx'][d_idx * self.batch_size: (d_idx + 1) * self.batch_size]
        return new_batch
    
    def save_best(self, val_results, i, save_path):
        if save_path:
            if self.best_records:
                comp = 0
                for task in val_results:
                    comp += (val_results[task] - self.best_records[task])/self.best_records[task]
                comp /= len(val_results)
            if not self.best_records or comp > 0:
                self.best_records = val_results
                self.save_model(i, save_path)
                print('saving to ' + save_path + ' ...', flush=True)
        return