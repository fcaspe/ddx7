import torch
from tqdm import tqdm
import h5py
import os.path as path
import os
import soundfile as sf
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import queue
import time

class ResultsLogger():
    def __init__(self,name,data_dir,sr):
        self.data_dir = data_dir
        self.filename = name
        self.step_count = 0
        self.store_dict = {}
        self.sr = sr
    def store_step(self,data):

        if(self.step_count == 0):
            for x in data:
                for k in x.keys():
                    if(k == 'audio' or k == 'synth_audio'):
                        data = x[k].squeeze(-1).reshape(-1)
                    else:
                        data = x[k]
                    self.store_dict[k] = data.detach().cpu().numpy()
        else:
            for x in data:
                for k in x.keys():
                    if(k == 'audio' or k == 'synth_audio'):
                        data = x[k].squeeze(-1).reshape(-1)
                        concat_dim = 0
                    else:
                        data = x[k]
                        concat_dim = -3
                    self.store_dict[k] = np.concatenate(
                                    (self.store_dict[k],data.detach().cpu().numpy()),
                                    axis=concat_dim)
                    #print("[DEBUG] ResultsLogger: Concat {}: {}".format(k,self.store_dict[k].shape))
        self.step_count += 1
        return

    def save_and_close(self):
        #print("[DEBUG] ResultsLogger: Saving {}.h5".format(self.filename))
        h5f = h5py.File(f'{self.data_dir}/{self.filename}.h5', 'w')
        #print("\t self.store_dict.keys() {}".format(self.store_dict.keys()))
        #print("\t self.store_dict['audio'] {}".format(self.store_dict['audio'].shape))
        #print("\t self.store_dict['synth_audio'] {}".format(self.store_dict['synth_audio'].shape))
        for k in self.store_dict.keys():
            h5f.create_dataset(k, data=self.store_dict[k])
        h5f.close()

        sf.write(
            path.join(self.data_dir, f"ref_{self.filename}.wav"),
            np.squeeze(self.store_dict['audio']),
            self.sr,
        )
        sf.write(
            path.join(self.data_dir, f'synth_{self.filename}.wav'),
            np.squeeze(self.store_dict['synth_audio']),
            self.sr,
        )
        self.step_count = 0
        self.store_dict = {}


class Hyperparams():
    def __init__(self,steps,loss_fn,opt,scheduler,lr,lr_decay_steps,lr_decay_rate,n_store_best,batch_size,test_loss_fn = None,grad_clip_norm=None):
        self.steps = steps
        self.loss_fn = loss_fn
        self.test_loss_fn = test_loss_fn
        self.opt = opt
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        self.grad_clip_norm = grad_clip_norm
        self.n_store_best = n_store_best

# Let's instantiate it using Hydra?
class Trainer():
    def __init__(self,loaders,preprocessor,
                hyperparams:Hyperparams,device):
        self.loaders = loaders
        self.preprocessor = preprocessor
        self.hp = hyperparams
        self.best_val_loss = np.inf
        self.device = device
        self.model = None
        self.writer = None
        self.opt = None
        #self.loss_fn = None
        self.scheduler = None
        self.train_step_counter = None
        self.best_model_epoch = None
        self.modelqueue = queue.SimpleQueue()
        self.epochs = self.hp.steps // len(loaders['train']) + 1
        return

    def register_stats(self,train_loss,val_loss,e):
        self.writer.add_scalar("train_loss", train_loss, e)
        self.writer.add_scalar("val_loss", val_loss, e)
        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], e)
        self.writer.add_scalar("reverb_decay", self.model.get_params('reverb_decay'), e)
        self.writer.add_scalar("reverb_wet", self.model.get_params('reverb_wet'), e)

    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def checkpoint(self,e):

        if(e==-1):
            torch.save(
                self.model.state_dict(),
                path.join("state_best.pth"),
            )
            return

        torch.save(
            self.model.state_dict(),
            path.join(f"state_{e:06d}.pth"),
        )
        self.best_model_epoch = e

        # Register new epoch to keep a trail of best models.
        self.modelqueue.put(e)
        # Delete a previous checkpoint if trail is bigger than indicated.
        if(self.modelqueue.qsize()>self.hp.n_store_best):
            delete_model_epoch = self.modelqueue.get()
            #print("[DEBUG] Deleting model from epoch: {}".format(delete_model_epoch))
            os.remove(
                path.join(f"state_{delete_model_epoch:06d}.pth")
            )
        return

    def load_checkpoint(self,model,e):
        if(self.device=='cpu'):
            if(e==-1):
                print("[INFO] Trainer.load_checkpoint on CPU: best")
                model.load_state_dict(
                    torch.load(
                        path.join(f"state_best.pth"),map_location=torch.device('cpu')
                ))
            elif(e!=0):
                print("[INFO] Trainer.load_checkpoint on CPU: {}".format(e))
                model.load_state_dict(
                    torch.load(
                        path.join(f"state_{e:06d}.pth"),map_location=torch.device('cpu')
                ))
        else:
            if(e==-1):
                print("[INFO] Trainer.load_checkpoint: best")
                model.load_state_dict(
                    torch.load(
                        path.join(f"state_best.pth")
                ))
            elif(e!=0):
                print("[INFO] Trainer.load_checkpoint: {}".format(e))
                model.load_state_dict(
                    torch.load(
                        path.join(f"state_{e:06d}.pth")
                ))
        return model

    def train(self):
        # The training loop
        for e in tqdm(range(self.epochs)):
            train_loss = self.train_step(self.model)
            val_loss = self.val_step(self.model)

            if self.hp.lr_decay_steps < self.train_step_counter:
                self.scheduler.step()
                self.train_step_counter = 0

            self.register_stats(train_loss,val_loss,e)

            if(self.best_val_loss > val_loss):
                self.best_val_loss = val_loss
                self.checkpoint(e)
        return

    def train_step(self,model):
        '''
        Run a train epoch
        '''
        model = model.train()
        mean_loss = 0
        nb = 0
        for x in self.loaders['train']:
            x = self.preprocessor.run(x)

            #print("INPUTS")
            #for k in x.keys():
            #    print(f'\t{k}: {x[k].size()} ')
            #print('')

            self.opt.zero_grad()
            synth_out = model(x)

            #print("OUTPUTS")
            #for k in synth_out.keys():
            #    print(f'\t{k}: {synth_out[k].size()} ')
            #print('')

            loss = self.hp.loss_fn(x['audio'],synth_out['synth_audio'])
            # TODO: Clip gradients
            loss.backward()

            if(self.hp.grad_clip_norm is not None):
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.hp.grad_clip_norm)

            self.opt.step()
            nb += 1
            mean_loss += (loss.item() - mean_loss) / nb
            self.train_step_counter += 1
            # Check for NaN during training
            if (torch.isnan(loss).any()):
                raise Exception("Warning - NaN detected in loss: {}".format(loss.item()))
        return mean_loss

    @torch.no_grad()
    def val_step(self,model):
        '''
        Run a validation epoch
        '''
        model = model.eval()
        mean_loss = 0
        nb = 0
        for x in self.loaders['valid']:
            x = self.preprocessor.run(x)

            synth_out = model(x)
            loss = self.hp.loss_fn(x['audio'],synth_out['synth_audio'])
            nb += 1
            mean_loss += (loss.item() - mean_loss) / nb
        return mean_loss


    @torch.no_grad()
    def test(self,model):
        '''
        Test the network over validation and test sets
        '''
        time.sleep(5) #Wait for all models to be written to disk.

        model = model.eval()
        for k in self.loaders.keys():
            if(k == 'train'): continue
            print(k)
            if(k == 'test_cnt'):
                # NOTE: Force test of long files on CPU
                # (there's an error on GPU rendering for long continuous instances.)
                # seems to be torch.cumsum according to comments online. TODO: Check this
                print(f'[INFO] running continuous rendering test on cpu. Enabling cumsum_nd() - rendering may be slow . . .')
                model = model.to('cpu')
                model.enable_cumsum_nd()
            mean_loss = 0
            nb = 0
            logger = ResultsLogger(f'{k}',
                                    data_dir='.',
                                    sr=model.get_sr())

            for x in self.loaders[k]:
                x = self.preprocessor.run(x)
                synth_out = model(x)
                # Check if we apply a special test fn for testing only
                if(self.hp.test_loss_fn is not None):
                    loss = self.hp.test_loss_fn(x['audio'],synth_out['synth_audio'])
                else:
                    loss = self.hp.loss_fn(x['audio'],synth_out['synth_audio'])
                nb += 1
                mean_loss += (loss.item() - mean_loss) / nb
                logger.store_step([x,synth_out])

            if(self.writer is not None):
                self.writer.add_scalar(f'final_{k}_loss', mean_loss)
            else:
                print('{} loss: {}'.format(k,mean_loss))
            logger.save_and_close()

        return mean_loss

    def run(self,model,mode='train',resume_epoch=None):
        torch.manual_seed(1234)
        self.model = model.to(self.device)
        n_params = self.count_parameters(self.model)
        print('[INFO] Model has {} trainable parameters.'.format(n_params))

        # Load checkpoint if needed (resume_epoch=-1 for best model)
        if(resume_epoch != 0):
            self.model = self.load_checkpoint(self.model,resume_epoch)
            # we should store and restore the state dict of scheduler and optimizer too.

        if(mode == 'train'):
            self.best_model_epoch = 0
            self.train_step_counter = 0
            self.writer = SummaryWriter(path.join('.'), flush_secs=20)
            self.opt = self.hp.opt(model.parameters(),lr=self.hp.lr)
            self.scheduler = self.hp.scheduler(self.opt,gamma=self.hp.lr_decay_rate)

            # Store number of parameters in tensorboard.
            self.writer.add_scalar("n_params", n_params )

            self.train()
            self.model = self.load_checkpoint(self.model,self.best_model_epoch)
            self.checkpoint(-1) #Store a separate copy of the best model.
            self.test(self.model)

        elif(mode == 'test'):
            self.test(self.model)

        return model
