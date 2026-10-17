import json

# Read the input JSON file
with open('results_optexp.json', 'r') as f:
    data = json.load(f)

# Extract the data for 4090 and fp16
matrix_data = data['4090']['fp16']

# Create the output structure
output = {}

# Process each matrix
for matrix_name, matrix_info in matrix_data.items():
    output[matrix_name] = {
        'dtc': matrix_info['64']['dtc'][0],
        'dtc-nobind': matrix_info['64']['dtc-nobind'][0]
    }

# Write the output to a new file
output_path = 'results_optexp_processed.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Processing complete. Results saved to {output_path}") 