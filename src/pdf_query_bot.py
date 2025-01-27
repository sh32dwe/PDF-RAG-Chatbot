import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class PDFQueryBot:
    def __init__(self, pdf_path, model_name="meta-llama/Llama-2-7b-chat-hf"):
        # Load environment variables
        load_dotenv()
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        
        if not hf_token:
            raise ValueError("Hugging Face token not found. Please set HUGGINGFACE_TOKEN in .env file")

        # PDF Loading and Text Splitting
        self.loader = PyPDFLoader(pdf_path)
        self.documents = self.loader.load()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.splits = self.text_splitter.split_documents(self.documents)
        
        # Embeddings and Vector Store
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = FAISS.from_documents(self.splits, self.embeddings)
        
        # LLM Initialization
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=hf_token,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        
        # Custom Prompt Template
        self.prompt_template = """
        Use the following context to answer the question.
        If the answer is not in the context, say "I cannot find the answer in the PDF".
        
        Context: {context}
        
        Question: {question}
        
        Helpful Answer:"""
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=['context', 'question']
        )
        
        # QA Chain
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=pipeline('text-generation', model=self.model, tokenizer=self.tokenizer),
            chain_type='stuff',
            retriever=self.retriever,
            chain_type_kwargs={'prompt': self.prompt}
        )

    def query(self, question):
        """Query the PDF with a question"""
        return self.qa_chain.run(question)

def main():
    # Example usage
    pdf_path = os.path.join('examples', 'sample.pdf')
    bot = PDFQueryBot(pdf_path)
    
    while True:
        question = input("Ask a question about the PDF (or type 'exit'): ")
        if question.lower() == 'exit':
            break
            
        try:
            answer = bot.query(question)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()