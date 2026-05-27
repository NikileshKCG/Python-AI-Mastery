"""
Project 4: File & Directory Organizer
Concepts: os, pathlib, shutil, datetime, argparse,
          Decorators, Generators, Context Manager,
          Recursive directory walking, Error Handling
"""

import os
import shutil
import json
import datetime
import argparse
from pathlib import Path            # Modern, OOP-style path handling


#----------------------------------------------------------------
# SECTION 1: File Type Categories
# Concept: Dict as a config/lookup table
#----------------------------------------------------------------

FILE_CATEGORIES = {
    "Images"        : [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".ico"],
    "Videos"        : [".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"],
    "Audio"         : [".mp3", ".wav", ".aac", ".flac", ".ogg", ".wma"],
    "Documents"     : [".pdf", ".doc", ".docx", ".txt", ".md", ".pptx", ".ppt", ".xls", ".xlsx"],
    "Code"          : [".py", ".js", ".ts", ".html", ".css", ".cpp", ".c", ".java", ".json", ".xml", ".yaml", ".yml"],
    "Archives"      : [".zip", ".tar", ".gz", ".rar", ".7z"],
    "Data"          : [".csv", ".tsv", ".db", ".sqlite", ".parquet", ".npy", ".pkl"],
    "Executables"   : [".exe", ".msi", ".sh", ".bat", ".apk"],
    "Fonts"         : [".ttf", ".otf", ".woff", ".woff2"],
}

def get_category(extension):
    """Return category name for a given file extension. """
    ext = extension.lower()
    for category, extensions in FILE_CATEGORIES.items():
        if ext in extensions:
            return category
    return "Others"


#------------------------------------------------------------
# SECTION 2: Decorator - Logger
# Concepts: Decorators wrap a function to add behavior
#           without changing the function itself.
#           Pattern: used heavily in Flask, FastAPI, PyTorch
#------------------------------------------------------------

def log_action(func):
    """ 
    Decorator that logs every file operation to a log file.
    @log_action above any function automatically wraps it
    """
    def wrapper(*args, **kwargs):           # *args/**kwargs pass through all arguments
        result = func(*args, **kwargs)
        #log to file
        log_entry = {
            "action"    : func.__name__,
            "timestamp" : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "args"      : str(args),
        }
        log_file = Path("organizer_log.json")
        logs = []
        if log_file.exists():
            with open(log_file, "r") as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
        logs.append(log_entry)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
        return result
    return wrapper

#--------------------------------------------------------------
# SECTION 3: Generator -- File Scanner
# Concepts: Generators use "yield" instaed of "return"
#           They produce items ONE AT A TIME - memory efficient
#           Critical for large datasets in AI/ML pipelines
#--------------------------------------------------------------

def scan_files(directory, recursive=False):
    """ 
    Generator that yields file Paths one by one.
    Uses 'yield' -- does NOT load all file into memory at once
    """
    path = Path(directory)

    if not path.exists():
        print(f" Directory not found: {directory}")
        return                  # stops the generator
    
    if recursive:
        # rglob('*') walks ALL subdirectories
        for item in path.rglob("*"):
            if item.is_file():
                yield item          #yiedl pauses and sends one item
    else:
        # glob('*') only scans the top level
        for item in path.glob("*"):
            if item.is_file():
                yield item

def scan_directories(directory):
    """ Generator that yields only subdirectory Paths"""
    path = Path(directory)
    for item in path.iterdir():
        if item.is_dir():
            yield item

#------------------------------------------------------------
# SECTION 4:Core File Operations
# Concepts: pathlib.Path methods, shutil, os
#           @log_action decorator applied here
#------------------------------------------------------------

@log_action
def copy_file(src, dst):
    """ Copy a file from src to dst. Creates dst dirs if needed. """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)       # mkdir with parents=True creates full path
    shutil.copy2(str(src), str(dst))                    # copy2 preserves metatdata
    print(f" Copied : {src.name} --> {dst}")

@log_action
def move_file(src, dst):
    """ Move a file from src to dst. """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f" Moved : {src.name} --> {dst}")

@log_action
def delete_file(path):
    """ Delete a single file"""
    p = Path(path)
    if p.exists() and p.is_file():
        p.unlink()
        print(f" Deleted: {p.name}")
    else:
        print(f" File not found: {path}")

@log_action
def rename_file(path, new_name):
    """ Rename a file within the same directory. """
    p = Path(path)
    new_path = p.parent / new_name          # / operator joins path in pathlib
    p.rename(new_path)
    print(f" Renamed: {p.name} --> {new_name}")

def create_directory(path):
    """ Create a directory (and any missing parents)."""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f" Created directory: {path}")

def delete_empty_dirs(directory):
    """ Walk and remove all empty subdirectories. """
    removed = 0
    #os.walk bottom-up so we remove deepest empty dirs first
    for dirpath, dirnames, filenames in os.walk(directory, topdown=False):
        if dirpath == directory:
            continue
        if not os.listdir(dirpath):         # listdir returns [] for empty dirs 
            os.rmdir(dirpath)
            print(f" Removed empty dir: {dirpath}")
            removed += 1
    print(f" Removed {removed} empty directory(ies).")

#-----------------------------------------------------------
# SECTION 5: Organizer -- Sort by type
# Concepts: Combinging generator + pathlib + shutil
#-----------------------------------------------------------

def organize_by_type(directory, move=True, recursive=False):
    """
    Organize files into subfolders by their type/extension.
    move=True --> move files 
    move=False --> copies files (non-destructive)
    """
    directory = Path(directory)
    moved = 0
    skipped = 0

    print(f"\n Organizing: {directory}")
    print(f" Mode: {'Move' if move else 'Copy'} | Recursive: {recursive} \n")

    for file_path in scan_files(directory, recursive):      #using our generator
        category = get_category(file_path.suffix)
        target_dir = directory / category
        target_path = target_dir / file_path.name

        # Skip if file is already in a category folder
        if file_path.parent == target_dir:
            skipped += 1
            continue

        # Handle duplicate filnames
        if target_path.exists():
            stem        = file_path.stem                # filename without extension
            suffix      = file_path.suffix
            timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            target_path = target_dir / f"{stem}_{timestamp}{suffix}"
        
        if move:
            move_file(file_path, target_path)
        else:
            copy_file(file_path, target_path)
        moved += 1

    print(f" \n Done! {moved} file(s) organized, {skipped} skipped.")

#----------------------------------------------------------
# SECTION 6: Organizer - Sort by Date
# Concept: datetime, file metadata (stat)
#----------------------------------------------------------

def organize_by_date(directory, move=True):
    """ Organize files into Year/Month SubFolders based on modification date. """
    directory = Path(directory)
    moved = 0

    for file_path in scan_files(directory):
        # stat() gives file metadata -- size, timestamps, permissions
        mod_time = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
        year     = str(mod_time.year)
        month    = mod_time.strftime("%m-%B")

        target_dir = directory / year / month
        target_path = target_dir / file_path.name

        if file_path.parent == target_dir:
            continue

        if move:
            move_file(file_path, target_path)
        else:
            copy_file(file_path, target_path)
        moved += 1

    print(f"\n {moved} file(s) organized by date. ")

#-----------------------------------------------------------
# SECTION 7: File Analytics 
# Concept: os.stat, sorted(), max(), aggregation
#-----------------------------------------------------------

def directory_report(directory, recursive=False):
    """ Analyze a directory and print a full report. """
    directory = Path(directory)

    if not directory.exists():
        print(f" Directory not found: {directory}")
        return
    
    type_count      = {}     # {category: count}
    type_size       = {}     # {category: total size in bytes}
    total_files     = 0
    total_size      = 0
    largest_file    = None
    largest_size    = 0

    for file_path in scan_files(directory, recursive):
        size        = file_path.stat().st_size      # file size in bytes
        category    = get_category(file_path.suffix)

        # dict.get(key, default) - if key missing return 0 
        type_count[category] = type_count.get(category, 0) + 1
        type_size[category]  = type.size.get(category, 0) + size
        total_files += 1
        total_size  += size

        if size > largest_size:
            largest_size = size
            largest_file = file_path
    
    if total_files == 0:
        print(f" No files Found")
        return
    
    print(f"""
            Directory Report: {directory.name}
---------------------------------------------------------------
Total Files     : {total_files}
Total Size      : {format_size(total_size)}
Largest File    : {largest_file.name if largest_file else 'N/A'} ({format_size(largest_size)})
-------------------------------------------------------------""")
    print(f" {'Category':<15} {'Files':>6} {'size':>10} {'Bar'}")
    print(" " + "-"*55)

    # Sort by file cont descending 
    for cat, count in sorted(type_count.items(), key=lambda x: x[1], reverse=True):
        size_str = format_size(type_size[cat])
        bar      = "S" * min(count, 30)         # cap bar at 30 chars
        print(f" {cat:<15} {count:>6} {size_str:>10} {bar}")

def format_size(size_bytes):
    """ Convert bytes to human-readable format ."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"

#--------------------------------------------------------------
# SECTION 8: Duplicate Funder
# Concepts: Hashing files, dict as a lookup map
#--------------------------------------------------------------

import hashlib

def get_file_hash(file_path):
    """ Return MD5 hash of a file's contents. """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:            # 'rb' = read binary
        # Read in chunks --- memroy efficient for large files
        for chunk in iter(lambda: f.read(65536), b""):          #iter with sentinel
            hasher.update(chunk)
    return hasher.hexdigest()

def find_duplicates(directory, recursive=False):
    """ Find duplicate files by comparing their MD5 hashes. """
    hash_map = {}       # {hash: [list of file paths with same hash]}

    print(f"\n Scanning for duplicates in {directory} \n")

    for file_path in scan_files(directory, recursive):
        try:
            file_hash = get_file_hash(file_path)
            if file_hash not in hash_map:
                hash_map[file_hash] = []
            hash_map[file_hash].append(file_path)
        except (PermissionError, OSError):
            pass        # skip files we can't read
    
    # Filter to only hashes with more then one file
    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    
    if not duplicates:
        print(" No duplicates found!!")
        return
    
    total_wasted = 0
    for file_hash, paths in duplicates.items():
        print(f" Duplicate group (hash: {file_hash[:8]}....):")
        for p in paths:
            size = p.stat().st_size
            print(f" {p} ({format_size(size)})")
        #wasted space = all copies minus the original
        wasted = p.stat().st_size * (len(paths) - 1)
        total_wasted += wasted

    print(f" \n Wasted space from duplicates: {format_size(total_wasted)}")

#------------------------------------------------------------
# SECTION 9: Bulk Rename
# Concept: enumerate, string formatting, pathlib
# -----------------------------------------------------------     

def bulk_rename(directory, prefix="file", start=1, extension_filter=None):
    """
    Rename all files in a directory to prefix_001, prefix_002, etc.
    extension_filter: only rename files with this extension (e.g. '.jpg')
    """
    directory = Path(directory)
    files     = sorted([f for f in directory.iterdir() if f.is_file()])

    if not files:
        print(f" No files to rename. ")
        return
    
    print(f"\n Bulk renaming {len(files)} file(s)...\n")
    for i, file_path in enumerate(files, start=start):
        new_name = f"{prefix}_{str(i).zfill(3)}{file_path.suffix}"  # zfill pads: 1 --> 001
        new_path = directory / new_name
        file_path.rename(new_path)
        print(f" {file_path.name} --> {new_name}")
    
    print(f"\n Renamed {len(files)} file(s).")

#----------------------------------------------------
# SECTION 10: View log
#----------------------------------------------------

def view_log():
    """ Dispaly the last 20 logged actions. """
    log_file = Path("organizer_log.json")
    if not log_file.exists():
        print(" No log file found.")
        return
    with open(log_file, "r") as f:
        logs = json.load(f)
    
    recent = logs[-20:]                     # slice -> last 20 entries
    print(f"\n Last {len(recent)} Actions:")
    print(" " + "-"*55)
    for entry in recent:
        print(f" [{entry['timestamp']}] {entry['actoin']}")

#----------------------------------------------------------
# SECTION 11: Main Menu
#-----------------------------------------------------------

def main():
    print("\n" + "="* 48)
    print(" FILE & DIRECTORY ORGANIZER ")
    print("="*48)

    while True:
        print("""
---------------- MAIN MENU ------------------------------------
              [1]  Organize Files by Type
              [2]  Organize Files by Date
              [3]  Directory Report
              [4]  Find Duplicate Files
              [5]  Bulk Rename Files
              [6]  Copy a File
              [7]  Move a File
              [8]  Delete a File
              [9]  Rename a File
              [10] Create Directory
              [11] Delete Empty Directories
              [12] View Action Log
              [0]  Exit
-------------------------------------------------------------""")
        
        choice = input(" Enter Choice: ").strip()

        try:
            if choice == "1":
                d = input(" Directory path: ").strip()
                m = input(" Move files? (yes/no, default yes): ").strip().lower()
                r = input(" Recursive? (yes/no, default no): ").strip().lower()
                organize_by_type(d, move=(m != "no"), recursive=(r == "yes"))
            
            elif choice == "2":
                d = input(" Directory path: ").strip()
                m = input(" Move files? (yes/no, default yes): ").strip().lower()
                organize_by_date(d, move=(m != "no"))
            
            elif choice == "3":
                d = input(" Directory path: ").strip()
                r = input(" Recursive? (yes/no): ").strip().lower()
                directory_report(d, recursive=(r == "yes"))
            
            elif choice == "4":
                d = input(" Directory path: ").strip()
                r = input(" Recursive? (yes/no): ").strip().lower()
                find_duplicates(d, recursive=(r == "yes"))
            
            elif choice == "5":
                d = input(" Directory path: ").strip()
                prefix = input(" Filename prefix: ").strip() or "file"
                ext = input(" Filter extension (e.g. .jpg, or blank for all): ").strip() or None
                bulk_rename(d, prefix=prefix, extension_filter=ext)
            
            elif choice == "6":
                src = input(" Source file path : ").strip()
                dst = input(" Destination file path: ").strip()
                copy_file(src, dst)
            
            elif choice == "7":
                src = input(" Source file path : ").strip()
                dst = input(" Destination file path: ").strip()
                move_file(src, dst)
            
            elif choice == "8":
                path = input(" File path: ").strip()
                confirm = input(f" Delete '{path}'? (yes/no): ").strip().lower()
                if confirm == "yes":
                    delete_file(path)
            
            elif choice == "9":
                path = input(" File path: ").strip()
                new_name = input("New name: ").strip()
                rename_file(path, new_name)
            
            elif choice == "10":
                path = input(" New directory path:").strip()
                create_directory(path)                
            
            elif choice == "11":
                d = input("Directory path: ").strip()
                delete_empty_dirs(d)
            
            elif choice == "12":
                view_log()
            
            elif choice == "0":
                print("\n GoodBye!!!\m")
                break

            else:
                print(" Invalid Choice, try again!")
        
        except FileNotFoundError as e:
            print(f" File not found: {e}")
        except PermissionError as e:
            print(f" Permission denied: {e}")
        except Exception as e:
            print(f" Error: {e}")

if __name__ == "__main__":
    main()