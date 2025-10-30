import os
import ast
import pandas as pd
from tqdm import tqdm
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.index import create_in, exists_in, open_dir
from whoosh.analysis import StemmingAnalyzer
import logging
import shutil
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_search_index(dataset_folder, index_folder, force_rebuild=False):
    """
    Membuat index pencarian dari dataset dengan validasi dan error handling
    
    Args:
        dataset_folder (str): Path ke folder dataset
        index_folder (str): Path ke folder index
        force_rebuild (bool): Paksa rebuild index meskipun sudah ada
    
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    print("\n" + "="*60)
    print("📌 MEMBUAT INDEX PENCARIAN")
    print("="*60 + "\n")
    
    # Validasi folder dataset
    if not os.path.exists(dataset_folder):
        print(f"❌ Folder dataset '{dataset_folder}' tidak ditemukan!")
        return False
    
    # Cek apakah index sudah ada
    if exists_in(index_folder) and not force_rebuild:
        response = input("⚠️  Index sudah ada. Rebuild? (y/n): ").lower()
        if response != 'y':
            print("✅ Menggunakan index yang sudah ada.")
            return True
        else:
            # Backup index lama
            backup_index(index_folder)
    
    # Buat folder index
    os.makedirs(index_folder, exist_ok=True)
    
    # Define schema dengan analyzer untuk stemming
    print("📝 Membuat schema index...")
    schema = Schema(
        doc_id=ID(stored=True, unique=True),
        title=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        source=TEXT(stored=True),
        content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        raw_title=STORED,  # Simpan judul asli tanpa stemming
        raw_content=STORED  # Simpan konten asli
    )
    
    # Buat index
    try:
        ix = create_in(index_folder, schema)
        logger.info(f"Index dibuat di: {index_folder}")
    except Exception as e:
        print(f"❌ Gagal membuat index: {e}")
        return False
    
    # Load dataset
    filepath = os.path.join(dataset_folder, "combined_stemmed_dataset.csv")
    
    if not os.path.exists(filepath):
        print(f"❌ File 'combined_stemmed_dataset.csv' tidak ditemukan di '{dataset_folder}'!")
        print("   Pastikan file dataset ada dan nama file sesuai.")
        return False
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset berhasil dimuat!")
        print(f"   📊 Jumlah dokumen: {len(df)}")
        print(f"   📋 Kolom: {', '.join(df.columns.tolist())}\n")
    except Exception as e:
        print(f"❌ Gagal membaca dataset: {e}")
        return False
    
    # Validasi kolom yang diperlukan
    required_columns = ['judul_tokens', 'konten_tokens', 'source']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ Kolom yang dibutuhkan tidak ada: {missing_columns}")
        return False
    
    # Indexing dengan batch processing untuk efisiensi
    print("🔄 Memulai indexing dokumen...\n")
    
    writer = ix.writer()
    success_count = 0
    error_count = 0
    
    try:
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Indexing", unit="doc"):
            try:
                # Parse tokens
                judul_list = safe_parse_tokens(row['judul_tokens'])
                konten_list = safe_parse_tokens(row['konten_tokens'])
                
                # Join tokens
                judul = " ".join(judul_list) if judul_list else ""
                konten = " ".join(konten_list) if konten_list else ""
                content = f"{judul} {konten}".strip()
                
                # Skip jika content kosong
                if not content:
                    logger.warning(f"Dokumen {i} kosong, dilewati")
                    error_count += 1
                    continue
                
                # Tambahkan ke index
                writer.add_document(
                    doc_id=f"doc_{i}",
                    title=judul if judul else "(Tanpa Judul)",
                    source=str(row['source']),
                    content=content,
                    raw_title=judul,
                    raw_content=konten[:500]  # Simpan preview saja
                )
                
                success_count += 1
                
                # Commit batch setiap 1000 dokumen untuk efisiensi
                if (i + 1) % 1000 == 0:
                    writer.commit()
                    writer = ix.writer()
                    logger.info(f"Batch commit: {i + 1} dokumen")
                
            except Exception as e:
                logger.error(f"Error pada dokumen {i}: {e}")
                error_count += 1
                continue
        
        # Final commit
        writer.commit()
        
        # Summary
        print("\n" + "="*60)
        print("✅ INDEXING SELESAI!")
        print("="*60)
        print(f"✅ Berhasil: {success_count} dokumen")
        if error_count > 0:
            print(f"⚠️  Gagal: {error_count} dokumen")
        print(f"📁 Index disimpan di: {index_folder}")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error fatal saat indexing: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        return False


def safe_parse_tokens(token_string):
    """
    Safely parse token string dengan error handling
    
    Args:
        token_string: String representasi list tokens
    
    Returns:
        list: List of tokens atau empty list jika gagal
    """
    try:
        if pd.isna(token_string):
            return []
        
        # Coba parse sebagai literal Python
        parsed = ast.literal_eval(str(token_string))
        
        if isinstance(parsed, list):
            return [str(token) for token in parsed if token]
        else:
            return []
    except (ValueError, SyntaxError) as e:
        logger.warning(f"Gagal parse tokens: {e}")
        # Fallback: split by space
        return str(token_string).split()
    except Exception as e:
        logger.error(f"Unexpected error parsing tokens: {e}")
        return []


def backup_index(index_folder):
    """
    Backup index yang sudah ada
    
    Args:
        index_folder (str): Path ke folder index
    """
    if not os.path.exists(index_folder):
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_folder = f"{index_folder}_backup_{timestamp}"
    
    try:
        shutil.copytree(index_folder, backup_folder)
        logger.info(f"Index di-backup ke: {backup_folder}")
        print(f"💾 Backup index dibuat: {backup_folder}")
    except Exception as e:
        logger.warning(f"Gagal backup index: {e}")


def get_index_stats(index_folder):
    """
    Mendapatkan statistik index
    
    Args:
        index_folder (str): Path ke folder index
    
    Returns:
        dict: Statistik index
    """
    if not exists_in(index_folder):
        return None
    
    try:
        ix = open_dir(index_folder)
        
        stats = {
            'doc_count': ix.doc_count(),
            'doc_count_all': ix.doc_count_all(),
            'last_modified': ix.last_modified(),
            'storage_used': sum(
                os.path.getsize(os.path.join(index_folder, f)) 
                for f in os.listdir(index_folder)
            ) / (1024 * 1024)  # MB
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        return None


def print_index_info(index_folder):
    """
    Menampilkan informasi tentang index
    """
    stats = get_index_stats(index_folder)
    
    if stats:
        print("\n" + "="*60)
        print("📊 INFORMASI INDEX")
        print("="*60)
        print(f"📁 Lokasi: {index_folder}")
        print(f"📄 Jumlah dokumen: {stats['doc_count']}")
        print(f"💾 Ukuran: {stats['storage_used']:.2f} MB")
        print(f"🕒 Terakhir diubah: {stats['last_modified']}")
        print("="*60 + "\n")
    else:
        print("❌ Index tidak ditemukan atau rusak!")