"""
/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/
"""

import sys
def replace_key_word(file_path):
    try:
        with open(file_path) as f:
            content = f.read()
        
        new_content = content.replace("__global__", "__attribute__((annotate(\"__global__\")))")

        new_file_path = "{}.pp".format(file_path)
        with open(new_file_path, 'w') as f:
            f.write(new_content)

    except IOError as e:
        print("error\n")


    print("preprocessing done!\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: Python <*.py> <file>")
        sys.exit(1)

    replace_key_word(sys.argv[1])

if __name__ == "__main__":
    main()
