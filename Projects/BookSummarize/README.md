## Book Summarization and Chatbot

Book: Chicken Soup For The Soul -  Chia Sẻ Tâm Hồn Và Quà Tặng Cuộc Sống
Link: https://thuviensach.vn/chicken-soup-for-the-soul-tap-1-chia-se-tam-hon-va-qua-tang-cuoc-song-8708.html

# Libraries
- langchain
- pinecone (can change to FAISS or Chroma)
- gpt-4o-mini
- langfuse

# How to run the project
1. Download pdf file
2. Change the path of file in `01.BookSummarize&Q_A.ipynb`
3. Run all the source code

Notice that:
- If you dont want to use the Langfuse, just comment code.

## Demo

Result from Langfuse
Question 1: Tác giả của cuốn sách là ai ? 
![alt text](<CleanShot 2024-07-21 at 23.54.49@2x.png>)

Question 2: Về tác giả và sự ra đời của cuốn sách ? 
![alt text](<CleanShot 2024-07-21 at 23.57.03@2x.png>)

Output from  program
![alt text](<CleanShot 2024-07-22 at 00.03.49@2x.png>)