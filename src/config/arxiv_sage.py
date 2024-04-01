class ARGS:
    dataset ='ogbn_arxiv'
    path = '/root/dataset/OGB/'
    num_parts = 10
    batch_size = 1
    xi = 0.1

    device = 0
    cc_num_workers = 4
    ss_num_workers = 0

    model = 'SAGE'
    use_bn = True
    num_layers = 3
    hidden_channels = 256
    dropout = 0.5
    lr = 0.001
    epochs = 80

    save_dir=''

