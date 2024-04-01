class ARGS:
    dataset ='Flickr'
    path = '/root/dataset/Flickr'
    num_parts = 4
    batch_size = 2
    xi = 0.1

    device = 0
    cc_num_workers = 4
    ss_num_workers = 0

    model = 'SAGE'
    use_bn = False
    num_layers = 3
    hidden_channels = 256
    dropout = 0.5
    lr = 0.001
    epochs = 60

    save_dir=''

