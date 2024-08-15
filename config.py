# choose what binary you want to generate the dataset
version = ["openssl-101f", "openssl-2c4", "openssl-6ca", "openssl-7a8", "openssl-9de",
           "openssl-9ea", "openssl-65", "openssl-120", "openssl-160", "openssl-651",
           "openssl-652", "openssl-653", "openssl-654", "openssl-655", "openssl-656",
           "openssl-657", "openssl-923", "openssl-962", "openssl-6455", "openssl-d23",
           "openssl-e4e", "openssl-ed5", "openssl-fad"]

arch = ["arm", "x86", "mips"]
compiler = ["gcc", "clang"]
optimizer = ["O0", "O1", "O2", "O3"]
dir_name = "data/extracted-acfg/"

# ABNeural
ABNeural_dataset_dir = "dataset/ABNeural/"
ABNeural_feature_size = 9
ABNeural_model_save_path = "model/ABNeural/Experiment_2/model_weight"
ABNeural_figure_save_path = "model/ABNeural/"

# some details about dataset generation
max_nodes = 500
min_nodes_threshold = 0
Buffer_Size = 1000
mini_batch = 10

# some params about training the network
learning_rate = 0.0001
epochs = 30
step_per_epoch = 5000
valid_step_pre_epoch = 3000
test_step_pre_epoch = 3000
T = 5
embedding_size = 64
embedding_depth = 2
