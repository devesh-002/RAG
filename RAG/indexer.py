import nltk
import asyncio
nltk.download('averaged_perceptron_tagger')
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    VectorStoreIndex,
)
import pandas as pd
from llama_index.core.agent import ReActAgent

from llama_index.core import SummaryIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.node_parser import MarkdownNodeParser
import os
from tqdm import tqdm
import pickle
# llm = Gemini(api_key=gemini_api_key, model_name="models/gemini-pro")
from llama_index.core import Document
import os
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from typing import Sequence, List
import json
import nest_asyncio  # noqa: E402
nest_asyncio.apply()


GROQ_API_KEY=""
llamaparse_api_key=""
pinecone_api=""
gemini_api_key=""
COHERE_API_KEY= ""

llm = Groq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
Settings.llm = llm
Settings.embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")


# gemini_embedding_model = GeminiEmbedding(api_key=gemini_api_key, model_name="models/embedding-001")

from pathlib import Path


# embed_model = gemini_embedding_model
async def build_agent_per_doc(nodes, file_base):
    print(file_base)


    vi_out_path = f"./data2/index/{file_base}"
    summary_out_path = f"./data2/index/{file_base}_summary.pkl"
    if not os.path.exists(vi_out_path):
        Path("./data2/index/").mkdir(parents=True, exist_ok=True)

        print(nodes[0])
        vector_index = VectorStoreIndex(nodes,embed_model=embed_model)
        vector_index.storage_context.persist(persist_dir=vi_out_path)
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=vi_out_path),
        )

    # build summary index
    summary_index = SummaryIndex(nodes)

    # define query engines
    vector_query_engine = vector_index.as_query_engine(llm=llm)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="compact", llm=llm
    )

    # extract a summary
    if not os.path.exists(summary_out_path):
        Path(summary_out_path).parent.mkdir(parents=True, exist_ok=True)
        summary = str(
            await summary_query_engine.aquery(
                "Extract a concise 1-2 line summary of this document"
            )
        )
        pickle.dump(summary, open(summary_out_path, "wb"))
    else:
        summary = pickle.load(open(summary_out_path, "rb"))

    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name=f"vector_tool_{file_base}",
                description=f"Useful for questions related to specific facts",
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name=f"summary_tool_{file_base}",
                description=f"Useful for summarization questions",
            ),
        ),
    ]

    # build agent
    function_llm =   Groq(model="llama3-8b-8192", api_key=GROQ_API_KEY) # Gemini(api_key=gemini_api_key, model_name="models/gemini-pro")

    agent=ReActAgent.from_tools(
    query_engine_tools,
    system_prompt=f""" \
 You are a specialized agent designed to answer queries about the `{file_base}.md` part of the SEC 10k filings.
 You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\

""",
    llm=function_llm,
    verbose=True,
)
    return agent, summary


async def build_agents(docs):
    node_parser = MarkdownNodeParser()

    # Build agents dictionary
    agents_dict = {}
    extra_info_dict = {}

    # # this is for the baseline
    # all_nodes = []

    for idx, doc in enumerate(tqdm(docs)):
        # nodes = node_parser.get_nodes_from_documents([doc])
        nodes = node_parser.get_nodes_from_documents([doc])
        # all_nodes.extend(nodes)

        # ID will be base + parent
        file_path = Path(doc.metadata["path"])
        file_base = str(file_path.parent.stem) + "_" + str(file_path.stem)
        agent, summary = await build_agent_per_doc(nodes, file_base)

        agents_dict[file_base] = agent
        extra_info_dict[file_base] = {"summary": summary, "nodes": nodes}
        # time.sleep(20)
    return agents_dict, extra_info_dict
async def work():
    agents_dict, extra_info_dict = await build_agents(docs)

if __name__ == '__main__':
        all_files_gen = Path("./data/").rglob("*")
        all_files = [f.resolve() for f in all_files_gen]

        all_html_files = [f for f in all_files if f.suffix.lower() == ".md"]
        

        docs = []
        for idx, f in enumerate(all_html_files):

            print(f"Idx {idx}/{len(all_html_files)}")
            # 
            # loaded_docs = reader.load_data(file=f, split_documents=True)
            loaded_docs = UnstructuredMarkdownLoader(f).load()
            # print(loaded_docs)
            # Hardcoded Index. Everything before this is ToC for all pages
            loaded_doc = Document(
                text="\n\n".join([d.page_content for d in loaded_docs]),
                metadata={"path": str(f)},
            )
            print(loaded_doc.metadata["path"])
            docs.append(loaded_doc)
        asyncio.run(work())
