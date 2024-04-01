class ARGS:
    dataset ='Reddit'
    path = '/root/dataset/Reddit'
    num_parts = 20
    batch_size = 1
    xi = 0.1

    device = 0
    cc_num_workers = 12
    ss_num_workers = 12

    model = 'GAT'
    num_layers = 3
    hidden_channels = 128
    dropout = 0.5
    lr = 0.001
    epochs = 60

    save_dir='../output'

