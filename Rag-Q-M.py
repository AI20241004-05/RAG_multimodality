# 第一支程式：建立多檔案的 Chroma 資料庫（PDF, PPTX, PPT）

#ffrom langchain.document_loaders import (PyMuPDFLoader, UnstructuredPowerPointLoader, 
#     UnstructuredWordDocumentLoader, TextLoader, CSVLoader, UnstructuredMarkdownLoader,
#     UnstructuredExcelLoader, UnstructuredHTMLLoader, UnstructuredFileLoader)
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
#import os
#import pytesseract
#from PIL import Image
#import speech_recognition as sr
#import moviepy.editor as mp
#from pptx import Presentation
#from transformers import AutoTokenizer
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
#vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding, collection_metadata={"mmap": True})
#retriever = vectordb.as_retriever() # 預設取回 4 筆資料
#Chroma 的檢索方式預設使用 最近鄰搜尋（KNN），如果 k 值設太大，容易拿到重複或過於相似的內容。
# 改用max_marginal_relevance（MMR）提高查詢多樣性
#fetch_k=70：先從 50 個候選文檔中選擇最具多樣性的 20 個
#retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 15, "fetch_k": 70, "score_threshold": 0.7})  # 取回 10 筆資料
#retriever = vectordb.as_retriever(search_kwargs={"k": 10, "return_score": True})   # 沒有參數return_score
retriever = vectordb.as_retriever(search_kwargs={"k": 10})   # 取回 10 筆資料

# 設定 LLM
llm = ChatOpenAI(openai_api_key='None', openai_api_base='http://127.0.0.1:1234/v1/')
#model="mistral-nemo-instruct-2407"  # 上行可增加指定模型名稱參數
#model="deepseek-coder-v2-lite-instruct"

# 建立基於檢索的問答系統
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    #chain_type="refine",  # 結果不好，也沒比較快
    #chain_type="map_reduce",  # 改成 map_reduce
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

# 查詢範例
queryE = "，根據提供的資料回答問題，不要使用你自己的知識。如果沒有足夠的資訊，就回答 '找不到其它相關資料'。" 
#queryE = "Answer the question based on the provided information only. Do not use your own knowledge. If there is not enough information, reply with 'No relevant information found.'"
#queryE = "，請在回答的每一個資訊點後面加上 `[來源: YOUR SOURCE]`，如果是你自己的知識，請標示 `[來源: LLM]`。"
#queryE="Please answer the question, and add [Source: YOUR SOURCE] after each piece of information. If the information comes from your own knowledge, mark it as [Source: LLM]."
#query = "有二個獨立問題，第一個是Alison Hawk 的工作是什麼和年齡是多少？第二個是Vue.JS又是誰開發的？Vue.JS和DOM差別？"
#query = "What is Alison Hawk's personality, motives and goals?"
#query = "Alison Hawk 的工作是什麼和年齡是多少？"
#query = "conda 和 pip的差別及用法？"
#query = "Google App Script是什麼?"
#query = "2330 2021/1/8 ~ 3/31最低價是多少?"
#query = "請說明2008年金融海嘯的經過？"
#query = "python class程式範例"
#query = "help指令"
#query = "將 Dart 與 Java 和 JavaScript 對比"
#query = "firebase講義在那個檔案？建立步驟是什麼？"
#query = "如何下載和使用firebase-auth-flutterfire-ui，要提供程式碼，以及如何修改pubspec.yaml"
query = "生成式AI倫理法律的學習資源"
#query = "The course of the June 4 Tiananmen event"
#query = "美國總統早餐吃什麼"
"""
#tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")  # 換成你用的模型
tokenizer = AutoTokenizer.from_pretrained("Mistral-Nemo-Instruct-2407")  # 換成你用的模型
#text = "這是一段測試文本，你可以試試看這是多少 tokens。"
tokens = tokenizer.encode(query + queryE)
print(f"Token 數量: {len(tokens)}")


import struct

with open("Mistral-Nemo-Instruct-2407-Q3_K_L.gguf", "rb") as f:
    data = f.read(1000)  # 讀取前 1000 Bytes 來找 meta 資訊

print(data[:500])  # 查看前 500 Bytes 內容
"""

result = qa.invoke(query + queryE)

# 顯示查詢結果及資料來源
print("查詢結果：")
print(result['result'])

print(f"{'='*30}\n資料來源：")
#print(retriever)
i=1
for doc in result['source_documents']:
#    print(f"來源{i:3d}: {doc.metadata.get('source', '未知')}，\n相似度分數: {doc.metadata.get('score', '未知')}，\n內容: {doc.page_content}\n")
    print(f"來源{i:3d}: {doc.metadata.get('source', '未知')}")
    print(f"相似度分數: {doc.metadata.get('score', '未知')}")
    page=doc.metadata.get('page', '未知')
    print(f"Page: {page + 1 if isinstance(page, int) else (int(page) + 1 if isinstance(page, str) and page.isnumeric() else page)}")
    print(f"Created Date: {doc.metadata.get('creationdate', '未知')}")
    print(f"Modified Date: {doc.metadata.get('moddate', '未知')}")
    print(f"內容: {doc.page_content}\n")
    #print(doc)
    i += 1
print("查詢結果：")
print(result['result'])
ct2 = datetime.datetime.now()
print(f"current time: {ct2}")

# ts store timestamp of current time
ts2 = ct2.timestamp()
print(f"{(ts2-ts)/60} min")
"""
for doc in result['source_documents']:
    print(f"來源: {doc.metadata.get('source', '未知')}，\n內容: {doc.page_content[:100]}...\n\n\n")
"""