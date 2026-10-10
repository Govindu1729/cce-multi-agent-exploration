"""
Kaggle GitHub Integration Script

This script allows Kaggle notebooks to directly clone and pull from your GitHub repository,
eliminating the need to manually zip and upload the codebase for every change.

Usage in Kaggle Notebook:
1. Add your GitHub Personal Access Token as a Kaggle Secret named "GITHUB_TOKEN".
2. Run this script at the top of your notebook.

For public repos, you can skip the token and just use the HTTPS URL.
"""

import os
import subprocess
import sys

def setup_repo(repo_url: str, branch: str = "main", use_secret: bool = True):
    """Clones or updates the GitHub repo in the Kaggle working directory."""
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    target_dir = f"/kaggle/working/{repo_name}"
    
    # Handle authentication for private repos
    auth_url = repo_url
    if use_secret and "GITHUB_TOKEN" in os.environ:
        token = os.environ["GITHUB_TOKEN"]
        auth_url = repo_url.replace("https://", f"https://{token}@")
    elif use_secret:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            token = user_secrets.get_secret("GITHUB_TOKEN")
            auth_url = repo_url.replace("https://", f"https://{token}@")
        except Exception:
            print("⚠️ GITHUB_TOKEN secret not found. Assuming public repo or token in env.")

    if os.path.exists(target_dir):
        print(f"🔄 Pulling latest changes for {repo_name} (branch: {branch})...")
        subprocess.run(["git", "-C", target_dir, "fetch", "origin"], check=True)
        subprocess.run(["git", "-C", target_dir, "checkout", branch], check=True)
        subprocess.run(["git", "-C", target_dir, "pull", "origin", branch], check=True)
    else:
        print(f"📥 Cloning {repo_name} (branch: {branch})...")
        subprocess.run(["git", "clone", "-b", branch, auth_url, target_dir], check=True)
        
    os.chdir(target_dir)
    print(f"✅ Working directory set to: {os.getcwd()}")
    
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "-q"], check=True)
    print("✅ Setup complete! Ready to train.")

if __name__ == "__main__":
    # TODO: Replace with your actual GitHub repo URL
    REPO_URL = "https://github.com/Govindu1729/cce-multi-agent-exploration.git"
    BRANCH = "main"  # Change to "agent/kaggle-integration" if testing the branch
    
    # Set use_secret=False if your repo is public
    setup_repo(REPO_URL, BRANCH, use_secret=True)
