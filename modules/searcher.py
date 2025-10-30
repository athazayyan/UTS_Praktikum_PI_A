import os
import pandas as pd
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def search_query(index_folder, query, top_k=5):
    
    if not query or query.strip() == "":
        return [], "❌ Query tidak boleh kosong!"
    
    if not os.path.exists(index_folder):
        return [], "❌ Index belum dibuat!"
    
    try:
        ix = open_dir(index_folder)
        results_list = []
        
        with ix.searcher() as searcher:
            parser = QueryParser("content", ix.schema)
            query_obj = parser.parse(query)
            
            whoosh_results = searcher.search(query_obj, limit=None)
            
            for hit in whoosh_results:
                results_list.append((
                    hit['title'], 
                    hit['source'], 
                    hit['content']
                ))
        
        if not results_list:
            return [], "❌ Tidak ada dokumen ditemukan!"
        
        vectorizer = CountVectorizer()
        
        doc_contents = [r[2] for r in results_list]  # content
        all_texts = [query] + doc_contents
        
        try:
            vectors = vectorizer.fit_transform(all_texts)
            
            cosine_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
            
        except Exception as e:
            return results_list[:top_k], f"⚠️  Pencarian selesai (tanpa ranking): {str(e)}"
        
        ranked_docs = []
        for i, doc in enumerate(results_list):
            ranked_docs.append((doc, cosine_scores[i]))
        
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_docs[:top_k], "✅ Pencarian selesai!"
    
    except Exception as e:
        return [], f"❌ Error: {str(e)}"