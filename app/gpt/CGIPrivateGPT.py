import os
import sys

import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredEmailLoader, UnstructuredPowerPointLoader, \
    Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Insert the directory
root_dir = "./app/gpt"


def run_model():
    print("Run Model")
    # defines for document reading and KnowledgeBase
    langchain.verbose = False
    embedding_model_name = "all-MiniLM-L6-v2"
    model_kwargs = {'device': 'gpu'}
    chunk_size = 500
    chunk_overlap = 50

    # Define the supported document extensions and corresponding loaders
    supported_extensions = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.msg': UnstructuredEmailLoader
    }
    # Embeddings for similarity search
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []

    # Load documents from the input folder
    for root, dirs, files in os.walk(f'{root_dir}/input/'):
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext.lower() in supported_extensions:
                file_path = os.path.join(root, filename)
                print("Loading document:", file_path)

                # Load the document using the appropriate document loader
                doc_loader = supported_extensions[ext.lower()](file_path)
                documents = doc_loader.load()
                text_chunks = text_splitter.split_documents(documents)
                docs.extend(text_chunks)

    # Create in-memory Qdrant instance
    knowledge_base = Qdrant.from_documents(
        docs,
        embeddings,
        location=":memory:",
        collection_name="doc_chunks",
    )
    '''
    # print and check how the similarity search work???

    user_input="PNC Virtual Wallet Student"
    context_docs = knowledge_base.similarity_search(user_input, k=3)

    context=""
    for doc in context_docs:
        if hasattr(doc, 'page_content'):
            page_content = doc.page_content
            context += page_content + " "  # Append the page_content to the text
    print(context)
    '''

    # ===================================== 20B Quantized LLM load ==========================
    # actual model loading
    model_id = "EleutherAI/gpt-neox-20b"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    device = "gpu"

    # tokenizer load
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # actual model load..
    model_4bit = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

    # system_prompt="Please analyze below context and give short and precise answer. \n Context : \n"

    system_prompt = "Please analyze below context and give detailed answer. \n Context : \n"
    return True


# extract the actual answer from the generated text.
def extract_reply(s):
    print('extract_reply')
    # Find the positions of the last occurrence of "<start>" and "<end>"
    start = "Answer:"
    end = "A:"
    start_pos = s.rfind(start) + len(start)
    end_pos = s.rfind(end)
    if end_pos < start_pos:
        end_pos = s.rfind("<|endoftext|>")
    if end_pos < start_pos:
        end_pos = len(s)
    # Extract the text between "<start>" and "<end>"
    if start_pos != -1 and end_pos != -1:
        text = s[start_pos:end_pos]
        return text
    else:
        return None
    # return s


def execute(user_input):
    print('extract_reply')
    # get the context from knowledgebase..
    docs = knowledge_base.similarity_search(user_input, k=3)
    context = ""
    for doc in docs:
        if hasattr(doc, 'page_content'):
            page_content = doc.page_content
            context += page_content + " "  # Append the page_content to the text

    # create prompt including context and user input..
    prompt = f"{system_prompt} " + context + "\n ------ \n Question: \n" + user_input + "\n Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # get the output from model
    outputs = model_4bit.generate(**inputs, max_new_tokens=128)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer_message = extract_reply(generated_text)
    # answer_message = extract_reply(user_input)
    return answer_message
