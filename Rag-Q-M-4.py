# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 13:57:08 2025
add streamlit
@author: user
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback 
import datetime
import csv
import os
import re

import tempfile
from pathlib import Path

from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

import streamlit as st

#TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
#LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

st.set_page_config(page_title="KMS")
st.title("KMS: Unleashing the Power of Knowledge and RAG")
    
def get_available_filename(base_filename):
    """
    檢查檔案是否存在，若存在則在檔名後加上 -1、-2...
    """
    filename = base_filename
    count = 1
    while os.path.exists(filename + ".csv"):
        filename = f"{base_filename}-{count}"
        count += 1
    return filename + ".csv"

def save_to_csv(results, filename="output"):
    filename = get_available_filename(filename)
    
    with open(filename, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(["來源", "相似度分數", "Page", "Created Date", "Modified Date", "內容"])
        
        for i, doc in enumerate(results['source_documents'], start=1):
            source = doc.metadata.get('source', '未知')
            score = doc.metadata.get('score', '未知')
            page = doc.metadata.get('page', '未知')
            page = page + 1 if isinstance(page, int) else (int(page) + 1 if isinstance(page, str) and page.isnumeric() else page)
            created_date = doc.metadata.get('creationdate', '未知')
            modified_date = doc.metadata.get('moddate', '未知')
            content = doc.page_content.replace("\n", " ")  # 移除換行符號，避免影響 CSV 格式
            
            writer.writerow([source, score, page, created_date, modified_date, content])
    
    print(f"查詢結果已儲存到: {filename}")

def embeddings_on_local_vectordb():
    # 設定向量資料庫的儲存路徑
    persist_directory = 'db'
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cuda'}  # 或 {'device': 'cpu'}，視硬體設備而定

    # 初始化嵌入模型
    embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    # 連接到已存在的 Chroma 資料庫
    #persist_directory 指定向量資料庫的儲存路徑
    #embedding_function 指定文字轉向量的模型，讓 Chroma 能夠對資料和查詢進行語意檢索
    #mmap:啟用記憶體映射，讓查詢更快，減少記憶體使用
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding, collection_metadata={"mmap": True})
    retriever = vectordb.as_retriever(search_kwargs={"k": 50})   # 取回 10 筆資料
    return retriever



# 查詢範例
WOLLM = True
queryWO = "，根據提供的資料回答問題，不要使用你自己的知識。如果沒有足夠的資訊，就回答 '找不到其它相關資料'。" 
queryW = "，請在回答的每一個資訊點後面加上 `[來源: YOUR SOURCE]`，如果是你自己的知識，請標示 `[來源: LLM]`。"
#query = "如何下載和使用firebase-auth-flutterfire-ui，要提供程式碼，以及如何修改pubspec.yaml"
#query = "What do American presidents eat for breakfast?"
#query = "有哪些在新店的喘息機構是可服務深坑?"
#query = "哪些建物受金融機構辦理不動產貨款規範"
#query = "What is Trump's tariff policy"
#query = "川普的關稅政策是什麼"
#query = "What is Trump's tariff policy"
#query = "What is Trump's tariff policy on Canada"
#query = "What is Warren Buffett's detailed comment on tariffs"
#query = "What is Alison Hawke's age and job？"
#query = "conda 和 pip的差別及用法？"

def query_llm(retriever, query):
    # 使用 with 語句來追蹤 tokens 使用量
    with get_openai_callback() as cb:
        # ct stores current time
        ct = datetime.datetime.now()

        # ts store timestamp of current time
        ts = ct.timestamp()
        
        # 設定 LLM
        llm = ChatOpenAI(openai_api_key='None', openai_api_base='http://127.0.0.1:1234/v1/')


        # 建立基於檢索的問答系統
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm = llm,
            retriever=retriever,            # 從向量資料庫中檢索相關內容
            return_source_documents=True,   # 回答時會返回參考的原始文件內容
        )
        result = qa_chain({'question': query + (queryWO if WOLLM else queryW), 'chat_history': st.session_state.messages})  # 提問 & 之前的對話紀錄
        st.session_state.messages.append((query, result['answer']))
        # result = qa.invoke(query + (queryWO if WOLLM else queryW))
        
        # 顯示查詢結果及資料來源
        print("查詢結果：")
        # print(result['result'])
        print(result['answer'])

        safe_filename = re.sub(r'[\/:*?"<>|]', '_', query)  # 取代非法字元
        save_to_csv(result, safe_filename + ("-WO" if WOLLM else "-W"))


        

        print(f"\n{'='*30}\nTokens 使用統計：")
        print(f"總 Tokens: {cb.total_tokens}")
        print(f"提示 Tokens: {cb.prompt_tokens}")
        print(f"完成 Tokens: {cb.completion_tokens}")



        ct2 = datetime.datetime.now()
        print(f"Start time: {ct}")
        print(f"End time: {ct2}")

        # ts store timestamp of current time
        ts2 = ct2.timestamp()
        print(f"Process lasted: {(ts2-ts)/60} min")
    return result['answer']

def boot():
    #
#    input_fields()
    #
#    st.button("Submit Documents", on_click=process_documents)
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []    
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if query := st.chat_input():
        st.chat_message("human").write(query)
        st.session_state.retriever = embeddings_on_local_vectordb()
        # if "retriever" in st.session_state:
        response = query_llm(st.session_state.retriever, query)
        # else:
            # response = query_llm_direct(query)

        st.chat_message("ai").write(response)

if __name__ == '__main__':
    #
    boot()