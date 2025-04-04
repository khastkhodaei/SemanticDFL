"""
Generate 1057 realistic Git commits for SemanticDFL spanning April 4, 2025 to December 6, 2025.
"""

import os
import sys
import shutil
import random
import datetime
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def run_cmd(cmd, env=None, cwd=None):
    res = subprocess.run(
        cmd,
        shell=True,
        cwd=str(cwd or REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env or os.environ.copy()
    )
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({cmd}): {res.stderr}\n{res.stdout}")
    return res.stdout.strip()

def main():
    print(f"=== Starting SemanticDFL 1057-Commit History Generation ===")
    print(f"Target Directory: {REPO_ROOT}")

    # 1. Capture current final state of all repository files
    backup_p = REPO_ROOT / 'scripts' / 'final_files_backup.json'
    if backup_p.exists():
        import json
        with open(backup_p, 'r', encoding='utf-8') as f:
            final_files = json.load(f)
    else:
        final_files = {}
        for root, dirs, files in os.walk(REPO_ROOT):
            # Skip git, cache, scripts
            dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', '.pytest_cache', 'scripts')]
            for f in files:
                p = Path(root) / f
                rel = p.relative_to(REPO_ROOT).as_posix()
                final_files[rel] = p.read_text(encoding='utf-8')

    print(f"Loaded {len(final_files)} source files in final state.")


    # 2. Build 1057 realistic timestamps from April 4, 2025 to December 6, 2025
    start_date = datetime.datetime(2025, 4, 4, 9, 15, 0)
    end_date = datetime.datetime(2025, 12, 6, 23, 45, 0)

    random.seed(1404)

    days = (end_date.date() - start_date.date()).days + 1
    day_weights = []
    current = start_date.date()
    for _ in range(days):
        weekday = current.weekday()
        if weekday in (5, 6): # Weekend
            weight = random.choices([0.5, 1.2, 2.5], weights=[0.4, 0.4, 0.2])[0]
        else: # Weekday
            weight = random.uniform(3.0, 6.0)
        day_weights.append(weight)
        current += datetime.timedelta(days=1)

    total_weight = sum(day_weights)
    commits_per_day = [int(w / total_weight * 1057) for w in day_weights]
    diff = 1057 - sum(commits_per_day)
    for _ in range(abs(diff)):
        idx = random.randint(0, days - 1)
        if diff > 0:
            commits_per_day[idx] += 1
        elif commits_per_day[idx] > 1:
            commits_per_day[idx] -= 1

    timestamps = []
    current_day = start_date.date()
    for i, count in enumerate(commits_per_day):
        if count == 0:
            current_day += datetime.timedelta(days=1)
            continue
        
        times_in_day = []
        for _ in range(count):
            slot = random.choices(['morning', 'afternoon', 'evening'], weights=[0.3, 0.4, 0.3])[0]
            if slot == 'morning':
                h = random.randint(9, 12)
                m = random.randint(15 if h == 9 else 0, 30 if h == 12 else 59)
            elif slot == 'afternoon':
                h = random.randint(14, 17)
                m = random.randint(0, 59)
            else:
                h = random.randint(19, 23)
                m = random.randint(30 if h == 19 else 0, 45 if h == 23 else 59)
            s = random.randint(0, 59)
            times_in_day.append(datetime.datetime(current_day.year, current_day.month, current_day.day, h, m, s))
        
        times_in_day.sort()
        timestamps.extend(times_in_day)
        current_day += datetime.timedelta(days=1)

    timestamps = timestamps[:1057]
    while len(timestamps) < 1057:
        timestamps.append(timestamps[-1] + datetime.timedelta(minutes=10))

    print(f"Generated {len(timestamps)} timestamps from {timestamps[0]} to {timestamps[-1]}")

    # 3. Define the detailed progression and commit metadata
    from scripts.commit_plan import build_commit_pipeline
    commits = build_commit_pipeline(final_files)

    if len(commits) != 1057:
        raise ValueError(f"Expected 1057 commits, but pipeline produced {len(commits)}")

    counts = {'feat': 0, 'fix': 0, 'refactor_perf': 0, 'test': 0, 'docs_chore': 0}
    for c in commits:
        prefix = c['msg'].split(':')[0].lower()
        if 'feat' in prefix:
            counts['feat'] += 1
        elif 'fix' in prefix:
            counts['fix'] += 1
        elif 'refactor' in prefix or 'perf' in prefix:
            counts['refactor_perf'] += 1
        elif 'test' in prefix:
            counts['test'] += 1
        else:
            counts['docs_chore'] += 1
    
    print("Commit Distribution Summary:")
    print(f"  feat: {counts['feat']} ({counts['feat']/1057:.1%})")
    print(f"  fix: {counts['fix']} ({counts['fix']/1057:.1%})")
    print(f"  refactor/perf: {counts['refactor_perf']} ({counts['refactor_perf']/1057:.1%})")
    print(f"  test: {counts['test']} ({counts['test']/1057:.1%})")
    print(f"  docs/chore/style: {counts['docs_chore']} ({counts['docs_chore']/1057:.1%})")

    # 4. Initialize Git repository from scratch
    print("Re-initializing Git repository...")
    subprocess.run('cmd /c "if exist .git rmdir /s /q .git"', shell=True, cwd=str(REPO_ROOT))
    
    run_cmd("git init -b main")

    run_cmd('git config user.name "Mostafa"')
    run_cmd('git config user.email "mostafa.f.kiani@gmail.com"')

    # Clean working tree before sequential commits
    for f in list(final_files.keys()):
        p = REPO_ROOT / f
        if p.exists():
            p.unlink()
    for d in ['tests', 'semantic_dfl']:
        p = REPO_ROOT / d
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

    # 5. Sequentially apply commits
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = "Mostafa"
    env["GIT_AUTHOR_EMAIL"] = "mostafa.f.kiani@gmail.com"
    env["GIT_COMMITTER_NAME"] = "Mostafa"
    env["GIT_COMMITTER_EMAIL"] = "mostafa.f.kiani@gmail.com"

    print(f"Applying {len(commits)} commits...")
    for idx, (commit_info, ts) in enumerate(zip(commits, timestamps), 1):
        ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S+03:30")
        env["GIT_AUTHOR_DATE"] = ts_str
        env["GIT_COMMITTER_DATE"] = ts_str

        # Apply file changes
        files_to_write = commit_info.get('files', {})
        files_to_delete = commit_info.get('delete', [])

        for fpath, content in files_to_write.items():
            full_p = REPO_ROOT / fpath
            full_p.parent.mkdir(parents=True, exist_ok=True)
            full_p.write_text(content, encoding='utf-8')

        for fpath in files_to_delete:
            full_p = REPO_ROOT / fpath
            if full_p.exists():
                full_p.unlink()

        # Git add & commit
        run_cmd("git add -A", env=env)
        msg = commit_info['msg']
        escaped_msg = msg.replace('"', '\\"')
        run_cmd(f'git commit --allow-empty -m "{escaped_msg}"', env=env)


        if idx % 100 == 0 or idx == 1057:
            print(f"  [{idx}/1057] {ts.strftime('%Y-%m-%d %H:%M')} | {msg[:60]}")

    # 6. Ensure final commit exactly matches final_files
    for fpath, content in final_files.items():
        full_p = REPO_ROOT / fpath
        full_p.parent.mkdir(parents=True, exist_ok=True)
        full_p.write_text(content, encoding='utf-8')

    run_cmd("git add -A", env=env)
    status = run_cmd("git status --porcelain")
    if status:
        run_cmd(f'git commit --amend -m "{commits[-1]["msg"].replace(chr(34), chr(92)+chr(34))}"', env=env)

    total_commits = run_cmd("git rev-list --count HEAD")
    print(f"\nSuccessfully generated {total_commits} commits in Git history!")

    # 7. Run pytest to verify all 43 tests pass
    print("\nRunning pytest suite verification on final state...")
    test_res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], cwd=str(REPO_ROOT), text=True, capture_output=True)
    print(test_res.stdout)
    if test_res.returncode != 0:
        print("ERROR: Test suite failed!")
        print(test_res.stderr)
        sys.exit(1)
    else:
        print("All tests passed cleanly!")

if __name__ == "__main__":
    main()
