import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline 

class PDFQueryBot:
    def __init__(self, pdf_path, model_name="gpt2"):  # Use a smaller model by default
        """
        Initialize the PDF Query Bot with a PDF and a Hugging Face model.

        Args:
            pdf_path (str): Path to the PDF file
            model_name (str): Hugging Face model name
        """
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

        # Initialize Hugging Face Pipeline
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, truncation=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        pipeline_model = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100
        )

        # Wrap the pipeline with LangChain's HuggingFacePipeline
        self.llm = HuggingFacePipeline(pipeline=pipeline_model)

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
            llm=self.llm,  # Use the LangChain-compatible LLM
            chain_type='stuff',
            retriever=self.retriever,
            chain_type_kwargs={'prompt': self.prompt}
        )

    def query(self, question):
        """
        Query the PDF for an answer to a specific question.

        Args:
            question (str): The question to query.

        Returns:
            str: The answer to the question.
        """
        return self.qa_chain.invoke(question)


def main():
    # Example usage
    pdf_path = os.path.join('examples', 'sample.pdf')
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")

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