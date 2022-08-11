class TrainGlobalConfig:
    ##train_setting
    cuda_id = '0'
    satellite = 'WorldView2'#datasets
    method = 'PMACNet'#model name
    train_batch_size = 32
    valid_batch_size = 32
    total_epochs = 200
    lr = 0.001
    lr_decay_freq = 50
    num_workers = 1
    resume = False
    ##model_setting
    ms_inp_ch = 8
    num_layers = 2
    latent_dim = 32
    seed = 52
    # 文件夹设置
