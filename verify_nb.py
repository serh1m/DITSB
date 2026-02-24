import json
import ast
import sys

def verify():
    print("Verifying Notebook Python cells syntax...")
    try:
        with open('llm_training/DITSB_Colab_Training.ipynb', 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Failed to load notebook: {e}")
        sys.exit(1)
        
    failed = False
    for i, cell in enumerate(nb.get('cells', [])):
        if cell['cell_type'] == 'code':
            source = "".join(cell.get('source', []))
            clean_source = ""
            for line in source.split('\n'):
                if line.strip().startswith('!') or line.strip().startswith('%'):
                    indent = len(line) - len(line.lstrip())
                    clean_source += ' ' * indent + 'pass\n'
                    continue
                clean_source += line + '\n'
                
            try:
                if clean_source.strip():
                    ast.parse(clean_source)
            except SyntaxError as e:
                print(f"Cell {i} Syntax Error on line {e.lineno}: {e.text}")
                failed = True
                
    if failed:
        print("Notebook validation failed.")
        sys.exit(1)
    else:
        print("Notebook validation passed: All Python syntax checks out.")
        
if __name__ == "__main__":
    verify()
