from pathlib import Path
def files_are_equal(file1: str, file2: str) -> bool:
    file1 = Path(file1)
    file2 = Path(file2)
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        return f1.read() == f2.read()

def first_diff_line(file1: str, file2: str) -> int | None:
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        line_num = 1

        for line1, line2 in zip(f1, f2):
            if line1 != line2:
                return line_num
            line_num += 1

        # Check if one file has extra lines
        if f1.readline() or f2.readline():
            return line_num

    return None  # files are identical


# Example usage:
line = first_diff_line("..\\BenchmarkDatasets\\NoAuction\\3.NoAuction_DecPre\\NoAuction_DecPre_Training\\Train_Dst_NoAuction_DecPre_CF_1.txt", "..\\BenchmarkDatasets\\NoAuction\\3.NoAuction_DecPre\\NoAuction_DecPre_Training\\Train_Dst_NoAuction_DecPre_CF_2.txt")
if line is None:
    print("Files are identical")
else:
    print(f"Files differ starting at line {line}")


# compare_files("..\\BenchmarkDatasets\\NoAuction\\3.NoAuction_DecPre\\NoAuction_DecPre_Training\\Train_Dst_NoAuction_DecPre_CF_1.txt", "..\\BenchmarkDatasets\\NoAuction\\3.NoAuction_DecPre\\NoAuction_DecPre_Training\\Train_Dst_NoAuction_DecPre_CF_2.txt")
"""
# Example usage
if files_are_equal("..\\BenchmarkDatasets\\NoAuction\\3.NoAuction_DecPre\\NoAuction_DecPre_Training\\Train_Dst_NoAuction_DecPre_CF_1.txt", "..\\BenchmarkDatasets\\NoAuction\\3.NoAuction_DecPre\\NoAuction_DecPre_Training\\Train_Dst_NoAuction_DecPre_CF_2.txt"):
    print("Files are identical")
else:
    print("Files are different")
"""