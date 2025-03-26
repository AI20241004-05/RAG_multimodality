# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:48:41 2025
列出 Chroma 向量資料庫中已儲存的資料
@author: user
"""

#from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 設定向量資料庫的儲存路徑
persist_directory = 'db'
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cuda'}  # 或 {'device': 'cpu'}，視硬體設備而定

# 初始化嵌入模型
embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# 連接到已存在的 Chroma 資料庫
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# 查詢所有已儲存的資料
# 使用空白查詢來獲取所有文件，n_results 設為足夠大的數字
#results = vectordb.similarity_search(query="", k=20000)  # 調整 k 值以匹配資料數量
all_docs = vectordb.get()
file = open("data.txt","w")
for i, doc in enumerate(all_docs['documents']):
    print(f"ID: {i}, 內容: {doc}")
    print(f"來源: {all_docs['metadatas'][i].get('source', '未知')}")
    file.write(f"ID: {i}, 內容: {doc}\n")
    file.write(f"來源: {all_docs['metadatas'][i].get('source', '未知')}\n")
    page = all_docs['metadatas'][i].get('page', '未知')
    pg = page + 1 if isinstance(page, int) else (int(page) + 1 if isinstance(page, str) and page.isnumeric() else page)
    print(f"Page: {pg}")
    file.write(f"Page: {pg}\n")
file.close()  
"""
file = open("data.txt","w")
# 列出目前資料庫中的文件
print("目前資料庫中的文件：")
for i, doc in enumerate(results):
    print(f"ID: {i}, 內容: {doc.page_content}")
    file.write(f"ID: {i}, \n內容: {doc.page_content}\n")
    print(f"來源: {doc.metadata.get('source', '未知')}")
    page=doc.metadata.get('page', '未知')
    pg = page + 1 if isinstance(page, int) else (int(page) + 1 if isinstance(page, str) and page.isnumeric() else page)
    #print(f"Page: {page + 1 if isinstance(page, int) else (int(page) + 1 if isinstance(page, str) and page.isnumeric() else page)}")
    print(f"Page: {pg}\n")
    file.write(f"來源: {doc.metadata.get('source', '未知')}\n")
    file.write(f"Page: {pg}\n")
file.close()    
"""