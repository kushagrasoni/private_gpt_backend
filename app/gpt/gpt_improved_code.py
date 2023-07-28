import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredEmailLoader, UnstructuredPowerPointLoader, \
    Docx2txtLoader

class DocumentLoader:
    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.msg': UnstructuredEmailLoader
    }

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def load_documents(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = []

        for root, dirs, files in os.walk(os.path.join(self.root_dir, 'input')):
            for filename in files:
                _, ext = os.path.splitext(filename)
                if ext.lower() in self.SUPPORTED_EXTENSIONS:
                    file_path = os.path.join(root, filename)
                    print("Loading document:", file_path)

                    # Load the document using the appropriate document loader
                    doc_loader = self.SUPPORTED_EXTENSIONS[ext.lower()](file_path)
                    documents = doc_loader.load()
                    text_chunks = text_splitter.split_documents(documents)
                    docs.extend(text_chunks)

        return docs

def run_model():
    print("Run Model")
    # Insert the directory
    root_dir = "./app/gpt"

    # Other configuration parameters
    embedding_model_name = "all-MiniLM-L6-v2"

    # Load documents from the input folder
    doc_loader = DocumentLoader(root_dir)
    docs = doc_loader.load_documents()

    # Embeddings for similarity search
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    # Create in-memory Qdrant instance
    knowledge_base = Qdrant.from_documents(
        docs,
        embeddings,
        location=":memory:",
        collection_name="doc_chunks",
    )

    system_prompt = "Please analyze below context and give detailed answer. \n Context : \n"
    return knowledge_base, docs, embeddings, system_prompt

def extract_reply(s):
    # Find the positions of the last occurrence of "<start>" and "<end>"
    start = "Answer:"
    end = "A:"
    start_pos = s.rfind(start) + len(start)
    end_pos = s.rfind(end)
    if end_pos < start_pos:
        end_pos = s.rfind("")
    if end_pos < start_pos:
        end_pos = len(s)
    # Extract the text between "<start>" and "<end>"
    if start_pos != -1 and end_pos != -1:
        text = s[start_pos:end_pos]
        return text
    else:
        return None

def execute(user_input, knowledge_base, docs, embeddings, system_prompt):
    # Load the language model only when needed during inference
    model_id = "EleutherAI/gpt-neox-20b"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    device = "cuda:0"

    # tokenizer load
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # actual model load..
    model_4bit = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

    # get the context from knowledgebase..
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
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
    # Garbage collection around CUDA
    if torch.cuda.is_available():
        print("GPU Garbage collection.....")
        torch.cuda.empty_cache()
    return answer_message

if __name__ == "__main__":
    knowledge_base, docs, embeddings, system_prompt = run_model()
    user_input = "PNC Virtual Wallet Student"
    answer_message = execute(user_input, knowledge_base, docs, embeddings, system_prompt)

    # Garbage collection around CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(answer_message)
