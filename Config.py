import os

class config:
    # data path
    root_path = os.getcwd()
    train_data_path =   os.path.join(root_path, 'data/ruby_train_0.jsonl')
    val_data_path   =   os.path.join(root_path, 'data/ruby_valid_0.jsonl')
    test_data_path  =   os.path.join(root_path, 'data/ruby_test_0.jsonl')

    # saved model path
    load_model_path = None

    # model relevant
    model_type = 'CodeBERT'
    models_dic = {
        'CodeBERT': 'microsoft/codebert-base',
        'CodeT5': 'Salesforce/codet5-base'
    }
    model_path = models_dic[model_type]

    # output dir
    output_dir = os.path.join(os.path.join(root_path, 'output'), model_type)

    # data processor
    max_source_length = 256
    max_target_length = 128

    # dataloader params
    train_params =  {'batch_size': 8, 'shuffle': True}
    val_params   =  {'batch_size': 8, 'shuffle': False}
    test_params  =  {'batch_size': 8, 'shuffle': False}

    # train args
    epoch               = 3
    num_train_epochs    = 3
    learning_rate       = 5e-5
    beam_size           = 10
    weight_decay        = 0.0
    adam_epsilon        = 1e-8
    max_grad_norm       = 1.0
    warmup_steps        = 0

    # other
    n_gpu = 1
    seed = 42
    
