This system is based on the following material then adds functions<br>
https://medium.com/@cch.chichieh/rag%E5%AF%A6%E4%BD%9C%E6%95%99%E5%AD%B8-streamlit-langchain-llama2-c7d1dac2494e <br>
chroma-D-all.py #purge chroma <br>
chroma-Query.py #lists chroma content <br>
Rag-multi.py  將檔案轉入chroma，chroma在程式目錄下的db <br>
Rag-Q-M.py     含來源和結果 <br>
Rag-Q-M-2.py 含來源、token和結果 <br>
Rag-Q-M-3.py  把來源存到檔案，和程式在同層目錄下 <br>
Rag-Q-M-4.py  含streamlit <br>
要載入檔案的資料夾固定在 C:\Users\user\Material\Report\source <br>
 <br>
Usage: <br>
conda create -n RAG_streamlit python=3.7 <br>
conda activate RAG_streamlit <br>
安裝套件pip install -r requirements.txt <br>
執行Rag-multi.py  將檔案轉入chroma <br>
streamlit run Rag-Q-M-4.py <br>
