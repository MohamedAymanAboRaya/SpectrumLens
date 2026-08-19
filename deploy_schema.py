"""
SpectrumLens — Schema Deployer
Deploys the database schema to Supabase.

Usage:
    python deploy_schema.py                    # Interactive — shows instructions
    python deploy_schema.py --sql              # Print the SQL to stdout (for copy-paste)
    python deploy_schema.py --url postgresql://...  # Deploy via direct connection
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
SQL_FILE = ROOT / "supabase_schema.sql"


def print_instructions():
    """Print clear deployment instructions."""
    sql_content = SQL_FILE.read_text()

    print("=" * 70)
    print("  SpectrumLens — Schema Deployment")
    print("=" * 70)
    print()
    print("  The Supabase database needs the schema deployed before")
    print("  the full app (app.py) can work.")
    print()
    print("  The OFFLINE DEMO (demo_app.py) works WITHOUT this step.")
    print()
    print("  To deploy, follow these steps:")
    print()
    print("  1. Open your Supabase Dashboard:")
    print(f"     https://supabase.com/dashboard/project/wydszhinfsgdyinlzhiz/sql")
    print()
    print("  2. Click 'New Query' (or use the SQL Editor)")
    print()
    print("  3. Paste the ENTIRE contents of supabase_schema.sql")
    print(f"     (File: {SQL_FILE})")
    print()
    print("  4. Click 'Run' (or press Ctrl+Enter)")
    print()
    print("  5. After the schema is deployed, upload your data:")
    print("     python day2_retrieval.py --upload")
    print()
    print("=" * 70)
    print()
    print("  Or run with --sql flag to print the SQL for copy-paste:")
    print("    python deploy_schema.py --sql | pbcopy   (macOS)")
    print("    python deploy_schema.py --sql | xclip    (Linux)")
    print()


def deploy_via_url(database_url: str) -> bool:
    """Deploy schema via direct PostgreSQL connection."""
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
        return False

    print(f"Connecting to database...")
    try:
        conn = psycopg2.connect(database_url, sslmode="require")
        conn.autocommit = True
        cur = conn.cursor()
        print("Connected!")
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

    sql = SQL_FILE.read_text()

    print(f"Executing schema from {SQL_FILE.name}...")
    try:
        cur.execute(sql)
        print("Schema deployed successfully!")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Note: {e}")
        else:
            print(f"Error: {e}")
            conn.close()
            return False

    # Verify
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'spectrumlens_clinical_chunks'
        );
    """)
    exists = cur.fetchone()[0]
    if exists:
        cur.execute("SELECT COUNT(*) FROM spectrumlens_clinical_chunks;")
        count = cur.fetchone()[0]
        print(f"  Table: spectrumlens_clinical_chunks ({count} rows)")
    else:
        print("  WARNING: Table not created. Check for errors above.")

    cur.execute("""
        SELECT routine_name FROM information_schema.routines
        WHERE routine_name IN ('match_clinical_chunks', 'hybrid_search_clinical_chunks', 'bm25_search_clinical_chunks')
        ORDER BY routine_name;
    """)
    funcs = [r[0] for r in cur.fetchall()]
    print(f"  Functions: {', '.join(funcs) if funcs else 'NONE'}")

    conn.close()
    print("\nDone! Now run: python day2_retrieval.py --upload")
    return True


def main():
    parser = argparse.ArgumentParser(description="Deploy SpectrumLens schema to Supabase")
    parser.add_argument("--sql", action="store_true", help="Print the SQL to stdout")
    parser.add_argument("--url", type=str, help="Direct database URL for deployment")
    args = parser.parse_args()

    if args.sql:
        print(SQL_FILE.read_text())
        return

    if args.url:
        success = deploy_via_url(args.url)
        sys.exit(0 if success else 1)

    # Check if DATABASE_URL is set
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        success = deploy_via_url(db_url)
        sys.exit(0 if success else 1)

    print_instructions()


if __name__ == "__main__":
    main()
