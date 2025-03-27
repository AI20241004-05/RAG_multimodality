chroma-D-all.py 清空chroma <br>
chroma-Query.py 列出chroma內容 <br>
Rag-multi.py  將檔案轉入chroma，chroma在程式目錄下的db
Rag-Q-M.py     含來源和結果
Rag-Q-M-2.py 含來源、token和結果
Rag-Q-M-3.py  把來源存到檔案，和程式在同層目錄下
Rag-Q-M-4.py  含streamlit
要載入檔案的資料夾固定在 C:\Users\user\Material\Report\source

Usage:
conda create -n RAG_streamlit python=3.7
conda activate RAG_streamlit
安裝套件pip install -r requirements.txt
執行Rag-multi.py  將檔案轉入chroma
streamlit run Rag-Q-M-4.py
