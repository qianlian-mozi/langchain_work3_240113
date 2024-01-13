import os 
from tqdm import tqdm
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma,FAISS
from langchain.document_loaders import PyPDFLoader,UnstructuredEPubLoader


# 将选用上述仓库中所有的 markdown、txt 文件作为示例语料库,为了将上述仓库中所有满足条件的文件路径找出来的目的所以做了如下函数。
def get_files(dir_path):
    # args：dir_path，目标文件夹路径
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        # os.walk 函数将递归遍历指定文件夹
        for filename in filenames:
            # 通过后缀名判断文件类型是否满足要求
            if filename.endswith(".md"):
                # 如果满足要求，将其绝对路径加入到结果列表
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".txt"):
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".pdf"):
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".epub"):
                file_list.append(os.path.join(filepath, filename))
    return file_list


# 使用 LangChain 提供的 FileLoader 对象来加载目标文件，得到由目标文件解析出的纯文本内容。
# 但是不同类型的文件需要对应不同的 FileLoader
# 所以需要使用分辨判断再分别调用对应的fileloader
def get_text(dir_path):
    # args：dir_path，目标文件夹路径
    # 首先调用上文定义的函数得到目标文件路径列表
    file_lst = get_files(dir_path)
    # docs 存放加载之后的纯文本对象
    docs = []
    # 遍历所有目标文件
    for one_file in tqdm(file_lst):
        file_type = one_file.split('.')[-1]
        if file_type == 'md':
            loader = UnstructuredMarkdownLoader(one_file)
        elif file_type == 'txt':
            loader = UnstructuredFileLoader(one_file)
        elif file_type == 'epub':
            loader = UnstructuredEPubLoader(one_file)
        elif file_type == 'pdf':
            loader = PyPDFLoader(one_file)
        else:
            # 如果是不符合条件的文件，直接跳过
            continue
        docs.extend(loader.load())
    return docs

# 使用上面两个函数
# tar_dir = [
#     "/root/data/vits/Bert-VITS2",
#     "/root/data/vits/Bert-VITS2-UI",
#     "/root/data/vits/ch_vits",
#     "/root/data/vits/chatgpt-vits-waifu",
#     "/root/data/vits/ChatWaifu",
#     "/root/data/vits/emotional-vits",
#     "/root/data/vits/fish-speech",
#     "/root/data/vits/MassTTS",
#     "/root/data/vits/PaddleSpeech",
#     "/root/data/vits/langchain_used",
    
#     "/root/data/vits/vits2_pytorch",
# ]

tar_dir = [
    # "/root/data/mental_cure/1.epub",
    "/root/data/det",
    # "/root/data/mental_cure/3.epub",
    # "/root/data/mental_cure/4.epub",
    # "/root/data/mental_cure/5.epub",
    # "/root/data/mental_cure/6.epub",
    # "/root/data/mental_cure/7.epub",
    # "/root/data/mental_cure/8.epub",
    # "/root/data/mental_cure/9.epub",
    # "/root/data/mental_cure",

]

docs = []
for dir_path in tar_dir:
    docs.extend(get_text(dir_path))



# 引入到 LangChain 框架中构建向量数据库。
# 由纯文本对象构建向量数据库，我们需要先对文本进行分块，接着对文本块进行向量化。
# chunk_size 每个文档的字符数量限制
# chunk_overlap: 两份文档重叠区域的长度
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=150, separators=["\n\n", "\n", "(?<\. )",  " ", ""]
                                                 )
split_docs = text_splitter.split_documents(docs)

# 选用开源词向量模型 Sentence Transformer 来进行文本向量化。
embeddings = HuggingFaceEmbeddings(model_name="/root/model/sentence-transformer")

# 选择 Chroma 作为向量数据库，基于上文分块后的文档以及加载的开源向量化模型，将语料加载到指定路径下的向量数据库：
# 定义持久化路径/向量数据库保存路径
persist_directory = './data_base/vector_det/Chroma'

# 加载数据库
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)

# vectordb = FAISS.from_documents(
#     documents=split_docs,
#     embedding=embeddings,
#     persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
# )
# 将加载的向量数据库持久化到磁盘上
vectordb.persist()