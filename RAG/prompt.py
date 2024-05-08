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
from llama_index.core.objects import (
    ObjectIndex,
    ObjectRetriever,
)
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.schema import QueryBundle
import pandas as pd
from llama_index.core.agent import ReActAgent
from llama_index.core import SummaryIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.node_parser import MarkdownNodeParser
import os
from tqdm import tqdm
import pickle
from llama_index.core import Document
import os
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from typing import Sequence, List
import json
from llama_index.core.agent import ReActAgent
import nest_asyncio  # noqa: E402
nest_asyncio.apply()


GROQ_API_KEY=""
llamaparse_api_key=""
pinecone_api=""
gemini_api_key=""
COHERE_API_KEY= ""


# llm = Gemini(api_key=gemini_api_key, model_name="models/gemini-pro")
llm = Groq(model="llama3-8b-8192", api_key=GROQ_API_KEY,device_map="auto")
Settings.llm = llm
Settings.embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")



# gemini_embedding_model = GeminiEmbedding(api_key=gemini_api_key, model_name="models/embedding-001")

from pathlib import Path


# embed_model = gemini_embedding_model
async def build_agent_per_doc(nodes, file_base):
    print(file_base)


    vi_out_path = f"./data/index/{file_base}"
    summary_out_path = f"./data/index/{file_base}_summary.pkl"
    if not os.path.exists(vi_out_path):
        Path("./data/index/").mkdir(parents=True, exist_ok=True)

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
    function_llm =   Groq(model="llama3-8b-8192", api_key=GROQ_API_KEY,device_map="auto") # Gemini(api_key=gemini_api_key, model_name="models/gemini-pro")

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
    return agents_dict, extra_info_dict

class CustomObjectRetriever(ObjectRetriever):
    def __init__(
        self,
        retriever,
        object_node_mapping,
        node_postprocessors=None,
        llm=None,
    ):
        self._retriever = retriever
        self._object_node_mapping = object_node_mapping
        self._llm = llm or Groq(model="llama3-8b-8192", api_key=GROQ_API_KEY,device_map="auto")
        self._node_postprocessors = node_postprocessors or []

    def retrieve(self, query_bundle):
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_str=query_bundle)

        nodes = self._retriever.retrieve(query_bundle)
        for processor in self._node_postprocessors:
            nodes = processor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )
        tools = [self._object_node_mapping.from_node(n.node) for n in nodes]

        sub_question_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=tools, llm=self._llm
        )
        sub_question_description = f"""\
Useful for any queries that involve comparing multiple documents. ALWAYS use this tool for comparison queries - make sure to call this \
tool with the original query. Do NOT use the other tools for any queries involving multiple documents.
"""
        sub_question_tool = QueryEngineTool(
            query_engine=sub_question_engine,
            metadata=ToolMetadata(
                name="compare_tool", description=sub_question_description
            ),
        )

        return tools + [sub_question_tool]

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
        agents_dict, extra_info_dict=asyncio.run(work())

        all_tools = []
        for file_base, agent in agents_dict.items():
            summary = extra_info_dict[file_base]["summary"]
            doc_tool = QueryEngineTool(
                query_engine=agent,
                metadata=ToolMetadata(
                    name=f"tool_{file_base}",
                    description=summary,
                ),
            )
            all_tools.append(doc_tool)


        obj_index = ObjectIndex.from_objects(
            all_tools,
            index_cls=VectorStoreIndex,
        )
        vector_node_retriever = obj_index.as_node_retriever(
            similarity_top_k=10,
        )

        custom_obj_retriever = CustomObjectRetriever(
            vector_node_retriever,
            obj_index.object_node_mapping,
            node_postprocessors=[CohereRerank(top_n=5,api_key="rC9L7bO1dCbwKXgcoJe0u9UGoXie5RRUcQKkznaY")],
            llm=llm,
        )


        top_agent = ReActAgent.from_tools(
            tool_retriever=custom_obj_retriever,
            system_prompt=""" \
        You are an agent designed to answer queries about the SEC filings.
        Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

        """,
            llm=Groq(model="llama3-8b-8192", api_key=GROQ_API_KEY,device_map="auto"),
            verbose=True,
        )

        """### Define Baseline Vector Store Index

        As a point of comparison, we define a "naive" RAG pipeline which dumps all docs into a single vector index collection.

        We set the top_k = 4
        """

        all_nodes = [
            n for extra_info in extra_info_dict.values() for n in extra_info["nodes"]
        ]

        base_index = VectorStoreIndex(all_nodes)
        base_query_engine = base_index.as_query_engine(similarity_top_k=4)

        """## Running Example Queries

        Let's run some example queries, ranging from QA / summaries over a single document to QA / summarization over multiple documents.
        """


        # response = top_agent.query(
        #     "What do you know about MSFT")
        while(1):
            query=input("Enter your query: ").strip()
            response = top_agent.query(query)
            print(str(response))
        # response = top_agent.chat(
        #     "What was the revenue growth of MSFT in 2015"
        # )
        # print(str(response))

# import requests

# url = 'https://api.example.com/endpoint'
# headers = {
#     'Authorization': 'Bearer AIzaSyAze56zJj0P-Tdt8Vrjcfzi19hpArgWz4A'  # Replace with your API key
# }

# try:
#     response = requests.get(url, headers=headers)

#     # Check the response status code
#     if response.status_code == 200:
#         print('Valid credentials. Successful connection!')
#     else:
#         print('Error in API call. Status code:', response.status_code)
#         print('Response:', response.text)

# except Exception as e:
#     print('Error during API call:', e)

# from llama_index.llms.gemini import Gemini
# os.environ["GOOGLE_API_KEY"] = "AIzaSyAze56zJj0P-Tdt8Vrjcfzi19hpArgWz4A"

# llm = Gemini()
# resp = llm.stream_complete(
#     "The story of Sourcrust, the bread creature, is really interesting. It all started when..."
# )

# # baseline
# response = base_query_engine.query(
#     "What was Microsofts revenue from 1995 to 2022. Explain",
# )
# print(str(response))

# response = top_agent.query(
#     "Compare the content in the agents page vs. tools page."
# )

# print(response)

# response = top_agent.query(
#     "Can you compare the compact and tree_summarize response synthesizer response modes at a very high-level?"
# )

# print(str(response))