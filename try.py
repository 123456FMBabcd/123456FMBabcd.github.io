from transformers import BertTokenizer, BertModel
import torch

#加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# 假设你的生成器模型在这里加载或定义
# generator = ...

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode_text_to_vector(text):
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0).to(device)
    with torch.no_grad():
        embeddings = bert_model(input_ids)[0]
    return embeddings.mean(dim=1)  # 使用BERT输出的嵌入向量的平均值作为文本向量

def generate_image(text):
    # 将输入的文本编码为向量
    text_vector = encode_text_to_vector(text)
    
    # 使用生成器生成图像
    generated_image = generator(text_vector)
    
    return generated_image
