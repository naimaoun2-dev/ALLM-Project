"""
Script to populate the Chroma DB with country information
"""
import os
from dotenv import load_dotenv
from rag_system import RAGSystem
from country_data import COUNTRIES_DATA, format_country_text

# Load environment variables
load_dotenv()

def populate_database():
    """Populate the database with country information"""
    print("Initializing RAG system...")
    try:
        rag_system = RAGSystem()
        print("[OK] RAG system initialized")
    except Exception as e:
        print(f"[ERROR] Error initializing RAG system: {str(e)}")
        return False
    
    print(f"\nPopulating database with {len(COUNTRIES_DATA)} countries...")
    
    success_count = 0
    for i, country in enumerate(COUNTRIES_DATA, 1):
        try:
            country_text = format_country_text(country)
            
            rag_system.add_user_document(
                text=country_text,
                title=f"Country: {country['country_name']}",
                metadata={
                    "source": "database",
                    "info_type": country["info_type"],
                    "country_name": country["country_name"],
                    "capital": country["capital"],
                    "continent": country["continent"],
                    "flag_colors": ", ".join(country["flag_colors"]),
                    "added_at": "pre-populated"
                }
            )
            print(f"[OK] [{i}/{len(COUNTRIES_DATA)}] Added: {country['country_name']}")
            success_count += 1
        except Exception as e:
            print(f"[ERROR] [{i}/{len(COUNTRIES_DATA)}] Error adding {country['country_name']}: {str(e)}")
    
    print(f"\n[SUCCESS] Successfully populated {success_count}/{len(COUNTRIES_DATA)} countries")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Country Database Population Script")
    print("=" * 50)
    populate_database()
    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)

