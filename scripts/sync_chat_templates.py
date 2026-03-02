"""
Sync chat templates from unsloth's chat_templates.py to our standalone file.

Usage:
    python scripts/sync_chat_templates.py [/path/to/unsloth/chat_templates.py]

Reads unsloth's chat_templates.py, extracts all CHAT_TEMPLATES entries
(no GPU/imports needed), and regenerates our standalone file.

If no path given, searches common venv locations automatically.
"""
import os
import re
import sys


def find_unsloth_chat_templates():
    """Try to find unsloth's chat_templates.py in known locations."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base, "..", "autotrain_env", "lib"),
        os.path.join(base, ".tmpvenv", "lib"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            for root, dirs, files in os.walk(c):
                if "unsloth" in dirs:
                    path = os.path.join(root, "unsloth", "chat_templates.py")
                    if os.path.isfile(path):
                        return path
    return None


def extract_templates(source_path):
    """Extract CHAT_TEMPLATES from unsloth source without imports."""
    with open(source_path) as f:
        source = f.read()

    # Strip everything before CHAT_TEMPLATES = {} and after the first def/class
    # that uses non-trivial imports
    lines = source.split("\n")

    # Find the CHAT_TEMPLATES = {} line
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "CHAT_TEMPLATES = {}":
            start = i
            break

    if start is None:
        print("ERROR: Could not find 'CHAT_TEMPLATES = {}' in source")
        return {}

    # Find where the template definitions end.
    # Skip small helper defs (like _ollama_template) that are part of the template
    # section. The real boundary is a function that uses heavy imports like
    # get_chat_template, test_chat_templates, etc.
    end = len(lines)
    # Known small helpers that live inside the template section
    template_section_helpers = {"_ollama_template"}
    for i in range(start + 1, len(lines)):
        stripped = lines[i]
        if stripped.startswith("def ") or stripped.startswith("class "):
            # Check if this is a known small helper
            func_name = stripped.split("(")[0].replace("def ", "").replace("class ", "").strip()
            if func_name in template_section_helpers:
                continue
            end = i
            break

    # Extract just the template assignment section
    template_section = "\n".join(lines[start:end])

    # Remove imports and small helper function definitions (they're stubbed in namespace)
    cleaned_lines = []
    skip_func = False
    for line in template_section.split("\n"):
        stripped = line.strip()
        if stripped.startswith("from ") or stripped.startswith("import "):
            continue
        # Skip helper function definitions that reference external modules
        if stripped.startswith("def _ollama_template"):
            skip_func = True
            continue
        if skip_func:
            # Skip the function body (indented lines after def)
            if stripped and not line[0].isspace():
                skip_func = False  # Back to module level
            else:
                continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)

    # Exec in a clean namespace with stubs for external references
    namespace = {
        "CHAT_TEMPLATES": {},
        "DEFAULT_SYSTEM_MESSAGE": {},
        "_ollama_template": lambda name: None,  # stub - we only need the jinja templates
        "OLLAMA_TEMPLATES": {},
        "__builtins__": __builtins__,
    }

    try:
        exec(cleaned, namespace)
    except Exception as e:
        print(f"Warning: exec error (partial extraction): {e}")

    templates = {}
    for key, val in namespace["CHAT_TEMPLATES"].items():
        if isinstance(val, tuple) and len(val) >= 1:
            templates[key] = val[0]
        elif isinstance(val, str):
            templates[key] = val

    return templates


def generate_standalone(templates, output_path):
    """Generate the standalone chat templates file."""
    lines = []
    lines.append('"""')
    lines.append("Standalone chat templates extracted from Unsloth.")
    lines.append("This file works without GPU dependencies.")
    lines.append("")
    lines.append("Source: Extracted from unsloth/chat_templates.py")
    lines.append(f"Total templates: {len(templates)}")
    lines.append("")
    lines.append("NOTE: These templates are used for LISTING available templates only.")
    lines.append("The actual message formatting uses:")
    lines.append("- CUDA: Unsloth's get_chat_template()")
    lines.append("- CPU/MPS: Tokenizer's native template via safe_apply_chat_template()")
    lines.append("")
    lines.append("To update, run: python scripts/sync_chat_templates.py")
    lines.append('"""')
    lines.append("")
    lines.append("# Chat templates dictionary")
    lines.append("CHAT_TEMPLATES = {}")
    lines.append("")

    for key in sorted(templates.keys()):
        tmpl = templates[key]
        lines.append(f"# {key} template")
        lines.append(f"CHAT_TEMPLATES[{key!r}] = (")
        lines.append(f"    {tmpl!r}")
        lines.append(")")
        lines.append("")

    # get_template_for_model
    lines.append("")
    lines.append("def get_template_for_model(model_name):")
    lines.append('    """')
    lines.append("    Get suggested template based on model name.")
    lines.append("")
    lines.append("    Args:")
    lines.append("        model_name: Name or path of the model")
    lines.append("")
    lines.append("    Returns:")
    lines.append("        Suggested template name or None")
    lines.append('    """')
    lines.append('    model_lower = model_name.lower() if model_name else ""')
    lines.append("")
    lines.append("    # Direct matches (order matters - more specific first)")
    lines.append('    if "llama-3" in model_lower or "llama3" in model_lower:')
    lines.append('        return "llama3"')
    lines.append('    elif "llama-2" in model_lower or "llama2" in model_lower:')
    lines.append('        return "llama"')
    lines.append('    elif "gemma-3n" in model_lower or "gemma3n" in model_lower:')
    lines.append('        return "gemma3n" if "gemma3n" in CHAT_TEMPLATES else "gemma-3n"')
    lines.append('    elif "gemma-3" in model_lower or "gemma3" in model_lower:')
    lines.append('        return "gemma3" if "gemma3" in CHAT_TEMPLATES else "gemma-3"')
    lines.append('    elif "gemma-2" in model_lower or "gemma2" in model_lower:')
    lines.append('        return "gemma2" if "gemma2" in CHAT_TEMPLATES else "gemma"')
    lines.append('    elif "gemma" in model_lower:')
    lines.append('        return "gemma"')
    lines.append('    elif "mistral" in model_lower:')
    lines.append('        return "mistral"')
    lines.append('    elif "phi-4" in model_lower or "phi4" in model_lower:')
    lines.append('        return "phi-4"')
    lines.append('    elif "phi-3.5" in model_lower or "phi-35" in model_lower:')
    lines.append('        return "phi-3.5" if "phi-3.5" in CHAT_TEMPLATES else "phi-35"')
    lines.append('    elif "phi-3" in model_lower or "phi3" in model_lower:')
    lines.append('        return "phi-3"')
    lines.append('    elif "qwen3.5" in model_lower or "qwen-3.5" in model_lower:')
    lines.append('        return "qwen3-instruct" if "qwen3-instruct" in CHAT_TEMPLATES else "qwen3"')
    lines.append('    elif "qwen2.5" in model_lower or "qwen-2.5" in model_lower:')
    lines.append('        return "qwen2.5"')
    lines.append('    elif "qwen3" in model_lower or "qwen-3" in model_lower:')
    lines.append('        return "qwen3"')
    lines.append('    elif "qwen" in model_lower:')
    lines.append('        return "qwen3"')
    lines.append('    elif "gpt-oss" in model_lower or "gptoss" in model_lower:')
    lines.append('        return "gpt-oss" if "gpt-oss" in CHAT_TEMPLATES else "gptoss"')
    lines.append("")
    lines.append("    return None")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Written {len(templates)} templates to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        source_path = sys.argv[1]
    else:
        source_path = find_unsloth_chat_templates()
        if source_path:
            print(f"Found unsloth at: {source_path}")
        else:
            print("Could not find unsloth/chat_templates.py. Provide path as argument.")
            sys.exit(1)

    if not os.path.isfile(source_path):
        print(f"File not found: {source_path}")
        sys.exit(1)

    templates = extract_templates(source_path)
    print(f"Extracted {len(templates)} templates:")
    for k in sorted(templates.keys()):
        print(f"  {k}")

    output = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "autotrain", "preprocessor", "chat_templates_standalone.py",
    )
    generate_standalone(templates, output)
