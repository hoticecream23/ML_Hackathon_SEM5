#!/usr/bin/env python
"""
Setup verification script for Hangman project.

Checks:
- Python version
- Required packages
- Directory structure
- Corpus file existence
- File permissions
"""

import sys
import os
import importlib

def check_python_version():
    """Check if Python version is 3.8+"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor} (requires 3.8+)")
        return False

def check_packages():
    """Check if required packages are installed"""
    print("\nChecking required packages...")
    
    required_packages = [
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('matplotlib', 'matplotlib'),
        ('sklearn', 'scikit-learn'),
        ('tqdm', 'tqdm'),
        ('hmmlearn', 'hmmlearn'),
        ('gym', 'gym'),
    ]
    
    all_installed = True
    for import_name, package_name in required_packages:
        try:
            importlib.import_module(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} (not installed)")
            all_installed = False
    
    return all_installed

def check_directory_structure():
    """Check if all required directories exist"""
    print("\nChecking directory structure...")
    
    required_dirs = [
        'data',
        'results',
        'src',
        'tests',
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (missing)")
            all_exist = False
    
    return all_exist

def check_source_files():
    """Check if all source files exist"""
    print("\nChecking source files...")
    
    required_files = [
        'src/utils.py',
        'src/hmm_model.py',
        'src/train_hmm.py',
        'src/hangman_env.py',
        'src/rl_agent.py',
        'src/train_rl.py',
        'src/evaluate_agent.py',
        'tests/test_env.py',
        'requirements.txt',
        'README.md',
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            all_exist = False
    
    return all_exist

def check_corpus():
    """Check if corpus file exists"""
    print("\nChecking corpus file...")
    
    corpus_path = 'data/corpus.txt'
    if os.path.isfile(corpus_path):
        # Check file size
        size = os.path.getsize(corpus_path)
        size_mb = size / (1024 * 1024)
        
        # Try to count lines
        try:
            with open(corpus_path, 'r') as f:
                num_lines = sum(1 for _ in f)
            print(f"  ✓ {corpus_path} ({size_mb:.2f} MB, {num_lines} lines)")
            
            if num_lines < 10000:
                print(f"  ⚠ Warning: Corpus has only {num_lines} words (expected ~50,000)")
                return True  # Still considered success
            
            return True
        except Exception as e:
            print(f"  ✗ Error reading corpus: {e}")
            return False
    else:
        print(f"  ✗ {corpus_path} (NOT FOUND)")
        print(f"     Please add a corpus.txt file with ~50,000 English words")
        return False

def main():
    """Run all checks"""
    print("=" * 60)
    print("Hangman Project Setup Verification")
    print("=" * 60 + "\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_packages),
        ("Directory Structure", check_directory_structure),
        ("Source Files", check_source_files),
        ("Corpus File", check_corpus),
    ]
    
    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✓ All checks passed! You're ready to start training.")
        print("\nNext steps:")
        print("  1. cd src")
        print("  2. python train_hmm.py")
        print("  3. python train_rl.py")
        print("  4. python evaluate_agent.py")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
    
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())