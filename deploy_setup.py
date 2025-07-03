#!/usr/bin/env python3
"""
Render Deployment Setup Script
Automatically prepares your MCP Document Analyzer for Render deployment
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr.strip()}")
        return False

def check_file_exists(file_path, description):
    """Check if a required file exists"""
    if os.path.exists(file_path):
        print(f"✅ {description} found: {file_path}")
        return True
    else:
        print(f"❌ {description} missing: {file_path}")
        return False

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Environment variables
.env

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Database backups
*.db.backup
*.sqlite.backup

# Temporary files
*.tmp
*.temp
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✅ .gitignore file created")

def main():
    """Main setup function"""
    print("🚀 MCP Document Analyzer - Render Deployment Setup")
    print("=" * 55)
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check required files
    print("\n📋 Checking required files...")
    required_files = [
        ('api_server.py', 'Flask API server'),
        ('document_analyzer.py', 'MCP Document Analyzer'),
        ('requirements_api.txt', 'API requirements'),
        ('Procfile', 'Render process configuration'),
        ('render.yaml', 'Render deployment config'),
        ('client_example.py', 'Client example')
    ]
    
    missing_files = []
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing required files: {', '.join(missing_files)}")
        print("Please ensure all required files are present before deployment.")
        return False
    
    # Create .gitignore if it doesn't exist
    print("\n📝 Setting up .gitignore...")
    if not os.path.exists('.gitignore'):
        create_gitignore()
    else:
        print("✅ .gitignore already exists")
    
    # Check if git is initialized
    print("\n🔧 Checking Git setup...")
    if not os.path.exists('.git'):
        print("📦 Initializing Git repository...")
        if not run_command('git init', 'Initialize Git repository'):
            return False
    else:
        print("✅ Git repository already initialized")
    
    # Check git status
    print("\n📊 Checking Git status...")
    run_command('git status --porcelain', 'Check Git status')
    
    # Add files to git
    print("\n📤 Adding files to Git...")
    if not run_command('git add .', 'Add all files to Git'):
        return False
    
    # Check if there are changes to commit
    result = subprocess.run('git diff --cached --quiet', shell=True)
    if result.returncode != 0:  # There are changes to commit
        print("\n💾 Committing changes...")
        commit_message = input("Enter commit message (or press Enter for default): ").strip()
        if not commit_message:
            commit_message = "Prepare for Render deployment"
        
        if not run_command(f'git commit -m "{commit_message}"', 'Commit changes'):
            return False
    else:
        print("✅ No changes to commit")
    
    # Check for remote origin
    print("\n🌐 Checking remote repository...")
    result = subprocess.run('git remote get-url origin', shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Remote origin found: {result.stdout.strip()}")
        
        # Ask if user wants to push
        push_choice = input("\n🚀 Push to remote repository? (y/n): ").strip().lower()
        if push_choice in ['y', 'yes']:
            if not run_command('git push origin main', 'Push to remote repository'):
                print("⚠️  Push failed. You may need to set up the remote repository first.")
    else:
        print("⚠️  No remote origin found")
        print("\n📝 Next steps:")
        print("1. Create a repository on GitHub")
        print("2. Add remote origin: git remote add origin <your-repo-url>")
        print("3. Push to GitHub: git push -u origin main")
    
    # Final checklist
    print("\n✅ Setup Complete! Final Checklist:")
    print("=" * 40)
    print("📋 Before deploying to Render:")
    print("   ✅ All required files are present")
    print("   ✅ Git repository is initialized")
    print("   ✅ Files are committed to Git")
    print("   ⚠️  Push to GitHub (if not done yet)")
    print("\n🚀 Ready for Render deployment!")
    print("\n📖 Next steps:")
    print("1. Push your code to GitHub (if not done)")
    print("2. Go to render.com and create account")
    print("3. Create new Web Service")
    print("4. Connect your GitHub repository")
    print("5. Follow the deployment guide in RENDER_DEPLOYMENT_GUIDE.md")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Setup completed successfully!")
        else:
            print("\n❌ Setup encountered errors. Please check the output above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)