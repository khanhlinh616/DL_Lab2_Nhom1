"""Deep fix for notebook outputs: Remove all local machine paths from outputs (not source)."""
import json, re, sys, os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def clean_output(out):
    """Clean output object: replace or remove local path strings."""
    changed = False
    text = out.get('text', None)
    if text is not None:
        raw = text if isinstance(text, str) else ''.join(text)
        
        if r'C:\Users' in raw or 'C:\\\\Users' in raw:
            # Drop entire FutureWarning outputs from TensorBoard
            if 'FutureWarning' in raw or 'site-packages' in raw:
                return None, True  # signal: drop this output
        
        # Replace project root path with generic placeholder
        if 'C:\\Project\\lab 2' in raw or 'C:\\\\Project\\\\lab 2' in raw:
            if isinstance(text, list):
                new_text = []
                for line in text:
                    line = line.replace('C:\\Project\\lab 2', '<PROJECT_ROOT>')
                    line = line.replace('C:\\\\Project\\\\lab 2', '<PROJECT_ROOT>')
                    new_text.append(line)
                if new_text != text:
                    changed = True
                out = dict(out)
                out['text'] = new_text
            elif isinstance(text, str):
                new_text = text.replace('C:\\Project\\lab 2', '<PROJECT_ROOT>')
                if new_text != text:
                    changed = True
                out = dict(out)
                out['text'] = new_text
    return out, changed

def fix_notebook_deep(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    total_changed = 0
    total_dropped = 0
    
    for cell in nb['cells']:
        outputs = cell.get('outputs', [])
        if not outputs:
            continue
        new_outputs = []
        for out in outputs:
            cleaned, changed = clean_output(out)
            if cleaned is None:
                total_dropped += 1
            else:
                new_outputs.append(cleaned)
                if changed:
                    total_changed += 1
        cell['outputs'] = new_outputs
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    return total_changed, total_dropped

main_nb = os.path.join(PROJECT_ROOT, 'notebooks', 'Lab2_Practice2_Main.ipynb')
if os.path.exists(main_nb):
    c, d = fix_notebook_deep(main_nb)
    print(f"Lab2_Practice2_Main.ipynb: {c} modified, {d} dropped")

# Verify what remains
for nb_path in [main_nb]:
    if not os.path.exists(nb_path):
        continue
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for i, cell in enumerate(nb['cells']):
        for out in cell.get('outputs', []):
            text = ''.join(out.get('text', [])) if isinstance(out.get('text', ''), list) else out.get('text', '')
            if r'C:\Users' in text:
                print(f"  REMAINING: {os.path.basename(nb_path)} cell {i}: {text[:100]}")
            if 'C:\\Project\\lab 2' in text:
                print(f"  REMAINING: {os.path.basename(nb_path)} cell {i}: {text[:100]}")
print("Done.")
