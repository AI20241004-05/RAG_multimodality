# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:59:27 2025
清空資料庫
@author: user
"""

#from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 設定資料庫路徑與模型
persist_directory = 'db'
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cuda'}

# 連接到 Chroma 資料庫
embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# 清空資料庫
vectordb.delete_collection()
print("✅ Chroma 資料庫已成功清空！")
