"""
Rewrite author and committer identity across all commits using git fast-export and fast-import (binary mode).
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

OLD_PATTERN = re.compile(rb'^(author|committer)\s+.*?\s+<.*?>\s+(\d+\s+[+-]\d{4})$', re.MULTILINE)
NEW_NAME_EMAIL = b"Mostafa Khastkhodaei <dev.khastkhodaei@gmail.com>"

def main():
    print("=== Rewriting Git Author and Committer Identity (Binary Mode) ===")
    
    # 1. Run git fast-export
    export_proc = subprocess.Popen(
        ["git", "fast-export", "--all"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    export_bytes, export_err = export_proc.communicate()
    if export_proc.returncode != 0:
        print(f"Error during fast-export: {export_err.decode('utf-8', errors='replace')}")
        sys.exit(1)
        
    print(f"Exported {len(export_bytes):,} bytes from fast-export stream.")
    
    # 2. Modify author and committer lines
    # Substitution replacement
    def repl(match):
        role = match.group(1)
        ts = match.group(2)
        return b"%s %s %s" % (role, NEW_NAME_EMAIL, ts)
        
    modified_bytes, replaced_count = OLD_PATTERN.subn(repl, export_bytes)
    print(f"Replaced {replaced_count} author/committer entries.")
    
    # 3. Create a temporary import into main branch
    import_proc = subprocess.Popen(
        ["git", "fast-import", "--force", "--quiet"],
        cwd=str(REPO_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    import_out, import_err = import_proc.communicate(input=modified_bytes)
    if import_proc.returncode != 0:
        print(f"Error during fast-import: {import_err.decode('utf-8', errors='replace')}")
        sys.exit(1)
        
    # Reset working tree to HEAD
    subprocess.run(["git", "reset", "--hard", "main"], cwd=str(REPO_ROOT), check=True)
    subprocess.run(["git", "checkout", "main"], cwd=str(REPO_ROOT), check=True)
    
    print("Fast-import complete!")
    
    # 4. Verify commit count and identities
    count = subprocess.check_output(["git", "rev-list", "--count", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
    print(f"Total Commits: {count}")
    
    authors = subprocess.check_output(["git", "log", "--format=%an <%ae>"], cwd=str(REPO_ROOT), text=True).strip().splitlines()
    unique_authors = set(authors)
    print(f"Unique Authors ({len(unique_authors)}): {unique_authors}")
    
    committers = subprocess.check_output(["git", "log", "--format=%cn <%ce>"], cwd=str(REPO_ROOT), text=True).strip().splitlines()
    unique_committers = set(committers)
    print(f"Unique Committers ({len(unique_committers)}): {unique_committers}")

if __name__ == "__main__":
    main()
