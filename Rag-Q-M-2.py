# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 15:23:38 2025
results to Screen
建立多檔案的 Chroma 資料庫（PDF, PPTX, PPT），原始資料檔置於 C:\\Users\\user\\Material\\Report\\source
@author: user
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback  # 加入這行來追蹤 tokens
import datetime

# ct stores current time
ct = datetime.datetime.now()
print(f"current time: {ct}")
# ts store timestamp of current time
ts = ct.timestamp()
# 設定向量資料庫的儲存路徑
persist_directory = 'db'
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cuda'}  # 或 {'device': 'cpu'}，視硬體設備而定

# 初始化嵌入模型
embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# 第二支程式：結合 RAG 進行查詢，並註明資料來源

# 連接到已存在的 Chroma 資料庫

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding, collection_metadata={"mmap": True})
retriever = vectordb.as_retriever(search_kwargs={"k": 200})   # 取回 10 筆資料

# 設定 LLM
llm = ChatOpenAI(openai_api_key='None', openai_api_base='http://127.0.0.1:1234/v1/')


# 建立基於檢索的問答系統
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

# 查詢範例
queryE = "，根據提供的資料回答問題，不要使用你自己的知識。如果沒有足夠的資訊，就回答 '找不到其它相關資料'。" 
#queryE = "，請在回答的每一個資訊點後面加上 `[來源: YOUR SOURCE]`，如果是你自己的知識，請標示 `[來源: LLM]`。"
#query = "如何下載和使用firebase-auth-flutterfire-ui，要提供程式碼，以及如何修改pubspec.yaml"
#query = "What do American presidents eat for breakfast?"
query = "有哪些在新店的喘息機構是可服務深坑?"
"""
result = qa.invoke(query + queryE)

# 顯示查詢結果及資料來源
print("查詢結果：")
print(result['result'])

print(f"{'='*30}\n資料來源：")
i=1
for doc in result['source_documents']:
    print(f"來源{i:3d}: {doc.metadata.get('source', '未知')}")
    print(f"相似度分數: {doc.metadata.get('score', '未知')}")
    page=doc.metadata.get('page', '未知')
    print(f"Page: {page + 1 if isinstance(page, int) else (int(page) + 1 if isinstance(page, str) and page.isnumeric() else page)}")
    print(f"Created Date: {doc.metadata.get('creationdate', '未知')}")
    print(f"Modified Date: {doc.metadata.get('moddate', '未知')}")
    print(f"內容: {doc.page_content}\n")
    i += 1
print("查詢結果：")
print(result['result'])
"""
# 使用 with 語句來追蹤 tokens 使用量
with get_openai_callback() as cb:
    result = qa.invoke(query + queryE)
    
    # 顯示查詢結果及資料來源
    print("查詢結果：")
    print(result['result'])
    
    print(f"{'='*30}\n資料來源：")
    i=1
    for doc in result['source_documents']:
        print(f"來源{i:3d}: {doc.metadata.get('source', '未知')}")
        print(f"相似度分數: {doc.metadata.get('score', '未知')}")
        page=doc.metadata.get('page', '未知')
        print(f"Page: {page + 1 if isinstance(page, int) else (int(page) + 1 if isinstance(page, str) and page.isnumeric() else page)}")
        print(f"Created Date: {doc.metadata.get('creationdate', '未知')}")
        print(f"Modified Date: {doc.metadata.get('moddate', '未知')}")
        print(f"內容: {doc.page_content}\n")
        i += 1
    print("查詢結果：")
    print(result['result'])
    
    # 顯示 tokens 使用量統計
    print(f"\n{'='*30}\nTokens 使用統計：")
    print(f"總 Tokens: {cb.total_tokens}")
    print(f"提示 Tokens: {cb.prompt_tokens}")
    print(f"完成 Tokens: {cb.completion_tokens}")
    print(f"總花費 (USD): ${cb.total_cost:.4f}")


ct2 = datetime.datetime.now()
print(f"current time: {ct2}")

# ts store timestamp of current time
ts2 = ct2.timestamp()
print(f"{(ts2-ts)/60} min")