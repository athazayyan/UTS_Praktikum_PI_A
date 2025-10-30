import sys
from modules.indexer import create_search_index
from modules.searcher import search_query

DATASET_FOLDER = "dataset"
INDEX_FOLDER = "index"


def menu():
    """Main menu loop"""
    while True:
        print("\n=== INFORMATION RETRIEVAL SYSTEM ===")
        print("[1] Load & Index Dataset")
        print("[2] Search Query")
        print("[3] Exit")
        print("====================================")
        
        choice = input("Pilih menu (ketik angka 1/2/3): ").strip()
        
        if choice == "1":
            # Menu 1: Load & Index Dataset
            print()
            create_search_index(DATASET_FOLDER, INDEX_FOLDER)
        
        elif choice == "2":
            # Menu 2: Search Query
            print("\n" + "="*60)
            print("🔍 MODE PENCARIAN")
            print("="*60)
            query = input("Query: ").strip()
            
            if not query:
                print("❌ Query tidak boleh kosong!")
                input("\nTekan Enter untuk kembali...")
                continue
            
            print("-"*60)
            print(f"Mencari: '{query}'...")
            print("-"*60)
            
            results, message = search_query(INDEX_FOLDER, query)
            
            print(message)
            
            if results:
                print("\n" + "="*60)
                print(f"📊 HASIL PENCARIAN (Top {len(results)})")
                print("="*60)
                
                for i, item in enumerate(results, 1):
                    try:
                        # Format dari searcher.py original: list of tuples
                        # Format bisa: ((title, source, content), score) atau (title, source, content, score)
                        
                        if isinstance(item, tuple):
                            if len(item) == 2:
                                # Format: ((title, source, content), score)
                                doc_data, score = item
                                if isinstance(doc_data, tuple) and len(doc_data) == 3:
                                    title, source, content = doc_data
                                else:
                                    # Fallback jika format berbeda
                                    title = str(doc_data)
                                    source = "Unknown"
                                    content = ""
                                    score = 0.0
                            elif len(item) == 4:
                                # Format: (title, source, content, score)
                                title, source, content, score = item
                            else:
                                print(f"[{i}] Format data tidak dikenali, skip...")
                                continue
                                
                        elif isinstance(item, dict):
                            # Format dictionary (jika menggunakan improved searcher)
                            title = item.get('title', '')
                            source = item.get('source', '')
                            content = item.get('content', '')
                            score = item.get('final_score', item.get('cosine_score', item.get('score', 0)))
                        else:
                            print(f"[{i}] Format tidak dikenali: {type(item)}")
                            continue
                        
                        # Display hasil dengan format rapi
                        print(f"\n[{i}] Relevance Score: {score:.4f}")
                        print(f"─" * 60)
                        
                        # Judul
                        if len(title) > 250:
                            print(f"📌 Judul : {title[:250]}...")
                        else:
                            print(f"📌 Judul : {title}")
                        
                        # Sumber
                        print(f"📚 Sumber: {source}")
                        
                        # Konten preview
                        if len(content) > 200:
                            print(f"📄 Konten: {content[:200]}...")
                        else:
                            print(f"📄 Konten: {content}")
                        
                        print("─" * 60)
                        
                    except Exception as e:
                        print(f"[{i}] Error menampilkan hasil: {e}")
                        continue
                
                print("\n💡 Tips: Gunakan query lebih spesifik untuk hasil lebih akurat")
            
            input("\n📌 Tekan Enter untuk kembali ke menu...")
        
        elif choice == "3":
            # Menu 3: Exit
            print("\n" + "="*60)
            print("👋 Terima kasih telah menggunakan sistem!")
            print("="*60 + "\n")
            sys.exit(0)


if __name__ == "__main__":
    try:
        menu()
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("⚠️  Program dihentikan oleh user (Ctrl+C)")
        print("="*60 + "\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error tidak terduga: {e}")
        print("\nDetail error:")
        import traceback
        traceback.print_exc()
        input("\nTekan Enter untuk keluar...")
        sys.exit(1)