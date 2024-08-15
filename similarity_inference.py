import os

import tensorflow as tf
import config
import pickle
import numpy as np
import random
from generate_dataset import zero_padded_adjmat,feature_vector

model = tf.keras.models.load_model(config.ABNeural_model_save_path)


class StrToBytes:
    def __init__(self, fileobj, mode):
        self.fileobj = fileobj
        self.mode = mode

    def read(self, size):
        if 'b' in self.mode:
            return self.fileobj.read(size)
        else:
            return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        if 'b' in self.mode:
            return self.fileobj.readline(size)
        else:
            return self.fileobj.readline(size).encode()

def read_acfg(filepath, out_path = "out_dir"):
    """
    读取IDA从binary中提取到的属性控制流图(ACFG)，并且转化成之后可操作的格式。
    filepath: ACFG的路径
    out_path: 对应输出文件的存储路径
    """
    all_function_dict = {}
    try:
        with open(filepath, "r") as f:
            picklefile = pickle.load(StrToBytes(f, "r"))
    except (UnicodeDecodeError, pickle.UnpicklingError):
        # 如果失败，尝试以二进制模式打开文件
        with open(filepath, "rb") as f:
            picklefile = pickle.load(StrToBytes(f, "rb"))
    for func in picklefile.raw_graph_list:
        if len(func.g) < config.min_nodes_threshold:
            continue
        if all_function_dict.get(func.funcname) is None:
            all_function_dict[func.funcname] = []
        all_function_dict[func.funcname].append(func.g)
    with open(out_path,"wb") as f:
        pickle.dump(all_function_dict,f)

def process_directory(directory, output_directory):
    """
    处理指定目录中的所有文件，调用 read_acfg 函数处理每个文件。
    directory: 要处理的文件所在目录
    output_directory: 转化后数据的输出目录
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            output_path = os.path.join(output_directory, filename)
            read_acfg(filepath, output_path)

def generate_embeddings(data_path = "out_dir"):
    """
    从转换后的ACFG数据文件中，读取每个函数的图属性信息，并使用DL将每个函数转化成一个embedding vector。
    Return:
        embeddings: 词典。 key 为函数名， value为对应的embedding vector。
    Note:
        embeddings的key可能需要根据情况自己设置，比如说，key应当设置为“当前软件名称”+ “exe/dll 文件名称" + ”函数名“。
        本函数实验中，仅使用函数名作为key，但这对于大规模项目来说是不合适，且不利于最后的检索的，请务必进行更改。
    """
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    embeddings = {}

    software_name = os.path.basename(data_path).split('.')[0]
    for funcname, graphs in data.items():
        for i,graph in enumerate(graphs):
            func_id = f"{software_name}_{funcname}_{i}"
            g_adjmat = zero_padded_adjmat(graph, config.max_nodes)
            g_featmat = feature_vector(graph, config.max_nodes)
            g_adjmat = np.expand_dims(g_adjmat,0)
            g_featmat = np.expand_dims(g_featmat,0)
            g_adjmat = tf.convert_to_tensor(g_adjmat)
            g_featmat = tf.convert_to_tensor(g_featmat)
            input = (g_adjmat, g_featmat, g_adjmat, g_featmat)
            sim, g1_embedding, g2_embedding = model(input)
            embeddings[func_id] = np.squeeze(g1_embedding.numpy())
    return embeddings

def save_embeddings(embeddings, save_file = "output/embeddings.txt"):
    """
    将生成得到的embedding存储到一个文件中去，方便之后进行比较。
    """
    with open(save_file, "w") as f:
        for key, value in embeddings.items():
            vector = value.tolist()
            vector_str = " ".join([str(v) for v in vector])
            f.write(key + "|" + vector_str + "\n")
#save_embeddings(embeddings, "output/embeddings.txt")

def process_output_directory(output_directory):
    """
    处理 output 目录中的所有 .cfg 文件，生成嵌入向量并保存到相应的文件中。
    output_directory: output 目录的路径
    """
    for filename in os.listdir(output_directory):
        if filename.endswith(".cfg"):
            filepath = os.path.join(output_directory, filename)
            embeddings = generate_embeddings(filepath)
            save_file = os.path.join(output_directory, filename.replace(".cfg", "_embeddings.txt"))
            save_embeddings(embeddings, save_file)
            """
            可以只保留生成嵌入文件
            """
            os.remove(filepath)

def load_embedding(file):
    """
    load embedding from a file.
    In the file, the embedding format is:
    func | embedding.

    :param file: filepath
    :return:
    """
    embedding_dict = {}
    with open(file, "r") as f:
        for line in f.readlines():
            if len(line) <= 1:
                continue
            sp = line.strip().split("|")
            funcname = sp[0]
            vector = sp[1:][0]
            vector = [float(v) for v in vector.split()]
            embedding_dict[funcname] = np.array(vector)
    return embedding_dict

def norm(vector):
    """
    l2 norm.
    :param vector:
    :return:
    """
    res = np.sqrt(np.sum(np.square(vector)))
    return vector / res

def infer_similarity(target_file, database_directory):
    """
    根据目标函数的嵌入向量，在 database_directory 目录中的所有嵌入文件中寻找相似的函数。
    target_file: 存储目标嵌入向量的文件。
    database_directory: 存储所有函数嵌入向量的目录。
    """
    target_embeddings = load_embedding(target_file)
    all_results = {}

    for target_funcname, target_embedding in target_embeddings.items():
        all_results[target_funcname] = []

        for filename in os.listdir(database_directory):
            if filename.endswith("_embeddings.txt"):
                embedding_file = os.path.join(database_directory, filename)
                my_embeddings = load_embedding(embedding_file)
                results = []

                for funcname, embedding in my_embeddings.items():
                    sim = np.dot(norm(target_embedding), norm(embedding))
                    if 0.98 <= sim <= 1.0:  # 设置阈值
                        results.append((funcname, sim))

                all_results[target_funcname].extend(results)

    # 使用写模式 ('w') 打开文件，确保每次运行都会覆盖之前的内容
    with open('../danger_result.txt', 'w', encoding='utf-8') as f:
        for target_funcname, sim_list in all_results.items():
            sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)[:10]
            if sim_list:
                f.write("********************\n")
                f.write(f"Similarity results for {target_funcname} as list below:\n")
                for funcname, sim in sim_list:
                    f.write(f"{funcname}: {sim}\n")
                f.write("\n")

    print("Results have been saved to danger_result.txt")

def random_test():
    my_embeddings = load_embedding("output/embeddings.txt")
    target_funcs = np.random.choice(list(my_embeddings.keys()), 10)
    for target_func in target_funcs:
        target_embedding = my_embeddings[target_func]
        infer_similarity(target_embedding, "output/embeddings.txt")
        print("--------------------")


if __name__ == "__main__":
    input_directory = "./target"
    output_directory = "./target/outs"
    process_directory(input_directory, output_directory)
    process_output_directory(output_directory)
    infer_similarity("./target/outs/jsonparse_embeddings.txt", "./data/database")

