class BaseConfig(): # GPU 1
    name = ''
    backbone = 'resnet50'
    classify = 'softmax'
    num_classes = 900
    metric = 'arc_margin'
    easy_margin = False
    loss = 'focal_loss'
    num_feature = 512
    pooling = None
    margin = 0.5

    display = True
    
    train_root = ''
    train_list = ''

    val_root = ''
    val_list = ''

    checkpoints_path = ''
    save_interval = 9
    
    load_model_url = ''
    load_model_path = ''
    load_margin_path = ''

    batch_size = 22  # batch size

    input_shape = (3,75,75) #  C, H, W     input_shape = (3,75, 500) #  C, H, W

    optimizer = 'sgd'
    num_workers = 32  # how many workers for loading data
    print_freq = 35  # print info every N batch
    val_freq = 2
    gpu = 0
    pretrained = False

    max_epoch = 100
    lr = 0.01 # initial learning rate
    lr_step = [40,70,90]
    lr_gamma = 0.1
    weight_decay = 1e-4
    momentum = 0.9

    def __getitem__(self, item):
        return self.__dict__[item]
