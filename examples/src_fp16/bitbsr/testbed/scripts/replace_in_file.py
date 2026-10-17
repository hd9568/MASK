import sys

def replace_in_file(input_file, search, replace, output_file):
    # Read the input file
    with open(input_file, 'r') as file:
        content = file.read()
    # Replace the search string with the replacement string
    content = content.replace(search, replace)
    # Write the modified content to the output file
    with open(output_file, 'w') as file:
        file.write(content)

if __name__ == "__main__":
    print(sys.argv)
    if len(sys.argv) != 5:
        print("Usage: python replace_in_file.py <input_file> <search> <replace> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    search = sys.argv[2]
    replace = sys.argv[3]
    output_file = sys.argv[4]
    
    replace_in_file(input_file, search, replace, output_file)