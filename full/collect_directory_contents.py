import os
import argparse
from pathlib import Path
import fnmatch

def parse_gitignore(gitignore_path):
    patterns = set()
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Bỏ qua comment và dòng trống
                if line and not line.startswith('#'):
                    # Chuẩn hóa pattern
                    if line.startswith('/'):
                        line = line[1:]
                    if line.endswith('/'):
                        line = line[:-1]
                    patterns.add(line)
    return patterns

def should_exclude(path, exclude_patterns, gitignore_patterns):
    # Kiểm tra các pattern từ command line
    if any(pattern in path for pattern in exclude_patterns):
        return True
    
    # Kiểm tra các pattern từ .gitignore
    path_parts = Path(path).parts
    for pattern in gitignore_patterns:
        # Xử lý pattern có dấu *
        if '*' in pattern:
            for part in path_parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
        # Xử lý pattern thông thường
        elif pattern in path_parts or pattern in path:
            return True
    return False

def collect_directory_contents(root_dir, output_file, exclude_patterns_folder, exclude_patterns_file):
    # Đọc .gitignore patterns
    gitignore_path = os.path.join(root_dir, '.gitignore')
    gitignore_patterns = parse_gitignore(gitignore_path)
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for root, dirs, files in os.walk(root_dir, topdown=True):
            # Loại bỏ các thư mục trong exclude_patterns_folder và .gitignore
            dirs[:] = [d for d in dirs if not should_exclude(d, exclude_patterns_folder, gitignore_patterns)]
            
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_dir)
                
                if not should_exclude(rel_path, exclude_patterns_file, gitignore_patterns):
                    out_file.write(f"File: {file_path}\n")
                    out_file.write("Content:\n")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as in_file:
                            content = in_file.read()
                            out_file.write(content)
                    except Exception as e:
                        out_file.write(f"Error reading file: {str(e)}\n")
                    out_file.write("\n" + "="*50 + "\n\n")

def main():
    parser = argparse.ArgumentParser(description="Collect directory contents into a single file.")
    parser.add_argument("-o", "--output", default="output.txt", help="Output file name (default: output.txt)")
    parser.add_argument("-efo", "--exclude-folder", nargs='+', default=[], help="Exclude patterns for folders")
    parser.add_argument("-efi", "--exclude-file", nargs='+', default=['.gitignore', 'collect_directory_contents.py', 'output.txt'], help="Exclude patterns for files")
    args = parser.parse_args()

    root_directory = os.getcwd()
    output_file = args.output
    exclude_patterns_folder = args.exclude_folder
    exclude_patterns_file = args.exclude_file

    collect_directory_contents(root_directory, output_file, exclude_patterns_folder, exclude_patterns_file)
    
    print(f"Đã hoàn thành. Kết quả được lưu trong {output_file}")
    print(f"Thư mục gốc: {root_directory}")
    print(f"Các mẫu loại trừ cho thư mục: {', '.join(exclude_patterns_folder)}")
    print(f"Các mẫu loại trừ cho file: {', '.join(exclude_patterns_file)}")

if __name__ == "__main__":
    main()