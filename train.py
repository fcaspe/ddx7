from trainer import Trainer
import torch
from ddx7.data_utils.h5_dataset import h5Dataset
from ddx7.data_utils.preprocessor import F0LoudnessRMSPreprocessor
from torch.utils.data import DataLoader, random_split
import hydra


def get_loaders(instrument,data_dir,train_split = 0.80,batch_size=16,device='cpu',load_additional_testset=True):

   # Loading Data and splitting it into train validation and test data
    print(hydra.utils.get_original_cwd())
    traindata_path = '{}{}'.format(hydra.utils.get_original_cwd(), f'/dataset/{data_dir}/train/{instrument}/16000.h5')
    input_keys = ('audio','loudness','f0','rms')
    traindset = h5Dataset(sr=16000,
                    data_path=traindata_path,
                    input_keys=input_keys,
                    max_audio_val=1,
                    device=device)

    train_split = int(train_split*len(traindset))
    test_split = (len(traindset)-train_split) // 2
    val_split = len(traindset)-train_split - test_split
    train, valid,test = random_split(traindset,[train_split,val_split,test_split])


    trainloader = DataLoader(train,
                            batch_size=batch_size,
                            shuffle = True,
                            drop_last=True)
    validloader = DataLoader(valid,
                            batch_size=batch_size,
                            shuffle = False,
                            drop_last=False) #Dont drop last for validation. Use all the data for validation.

    testloader = DataLoader(test,
                            batch_size=batch_size,
                            shuffle = False,
                            drop_last=False) #Dont drop last for test. Use all the data for test.


    print('[INFO] Train Dataset len: {}'.format(len(train)))
    print('[INFO] Valid Dataset len: {}'.format(len(valid)))
    print('[INFO] Test Dataset len: {}'.format(len(test)))
    loaders = {}
    loaders['train'] = trainloader
    loaders['valid'] = validloader
    loaders['test'] = testloader

    if(load_additional_testset == True):
        # Now load test with additional, non-URMP, audio excerpts.
        testcdata_path = '{}{}'.format(hydra.utils.get_original_cwd(), f'/dataset/{data_dir}/test/{instrument}/16000.h5')
        test_cnt = h5Dataset(sr=16000,
                        data_path=testcdata_path,
                        input_keys=input_keys,
                        max_audio_val=1,
                        device='cpu') #Force test set on CPU

        testcloader = DataLoader(test_cnt,
                                batch_size=1, # Always batch size of 1 for continuous test
                                shuffle = False, #don't shuffle for test set.
                                drop_last=False)

        print('[INFO] Test Continuous Dataset len: {}'.format(len(test_cnt)))
        loaders['test_cnt'] = testcloader

    return loaders

@hydra.main(config_path="recipes",config_name="config.yaml")
def main(args):
    torch.manual_seed(args.seed)

    loaders = get_loaders(args.instrument,args.data_dir,train_split=args.train_split,
        batch_size=args.hyperparams.batch_size,device=args.device,
        load_additional_testset=args.load_additional_testset)

    hyperparams = hydra.utils.instantiate(args.hyperparams)
    # After instantiation, adjust hyperparam config (will not need this in Hydra 1.2)
    if hyperparams.opt == 'Adam':
        hyperparams.opt = torch.optim.Adam
    if hyperparams.scheduler == 'ExponentialLR':
        hyperparams.scheduler = torch.optim.lr_scheduler.ExponentialLR

    model = hydra.utils.instantiate(args.model)
    trainer = Trainer(loaders=loaders,
                    preprocessor=F0LoudnessRMSPreprocessor(),
                    hyperparams=hyperparams,
                    device = args.device)

    trainer.run(model,
                mode=args.mode,
                resume_epoch=args.resume_epoch)

if __name__ == "__main__":
    main()