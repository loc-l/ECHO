class ARGS:
    dataset ='ogbn_products'
    path = '/root/dataset/'
    num_parts = 1000
    batch_size = 50
    xi = 0.1

    device = 0
    cc_num_workers = 12
    ss_num_workers = 12

    model = 'SAGE'
    use_bn = False
    num_layers = 3
    hidden_channels = 256
    dropout = 0.5
    lr = 0.001
    epochs = 70

    save_dir='../output'

