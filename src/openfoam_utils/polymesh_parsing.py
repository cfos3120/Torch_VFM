import re
import numpy as np

# Parse Owner/Neighbour Files
def parse_owner_file(path):
    with open(path, "r") as f:
        text = f.read()

    # Remove C++ // comments
    text = re.sub(r"//.*", "", text)

    # Remove the header
    brace_match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not brace_match:
        raise ValueError("No top-level { } block found")

    text = text[brace_match.end():]

    # Extract count and list
    # Find the first integer (entry count)
    count_match = re.search(r"\b(\d+)\b", text)
    if not count_match:
        raise ValueError("No entry count found")

    expected_count = int(count_match.group(1))
    remainder = text[count_match.end():]

    # Find the parenthesized list
    paren_match = re.search(r"\((.*?)\)", remainder, flags=re.DOTALL)
    if not paren_match:
        raise ValueError("No ( ... ) list found")

    list_body = paren_match.group(1)

    # Parse 
    values = [int(x) for x in re.findall(r"-?\d+", list_body)]

    # Validate
    if len(values) != expected_count:
        raise ValueError(f"Count mismatch: header says {expected_count}, but found {len(values)} entries")

    return values

def parse_openfoam_face_values(filename, field_type='auto', internal_only=False):
    """
    Parse OpenFOAM field file and extract ALL data from internalField, 
    concatenating multiple bracket sets if they exist.
    
    Args:
        filename: OpenFOAM field file path
        field_type: 'scalar', 'vector', or 'auto' 
        internal_only: If True, take only first block after count
    
    Returns:
        list of scalars OR list of (x,y,z) tuples

    NOTE: This ignores empty patches in concatenation
    """
    with open(filename, 'r') as f:
        content = f.read()
    
    # Find internalField section
    pattern = r'internalField\s+nonuniform\s+List<(\w+)>\s*\n(\d+)\s*\((.*)\)\s*;'
    match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
    
    if not match:
        raise ValueError("No internalField nonuniform List<scalar/vector> found")
    
    list_type = match.group(1)
    n_items = int(match.group(2))
    full_data_block = match.group(3)
    
    if field_type == 'auto':
        field_type = list_type
    
    if field_type == 'scalar':
        # Extract ALL numbers from entire block (handles multiple () sets)
        all_values = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', full_data_block)
        result = [float(v) for v in all_values]
        
        if internal_only:
            result = result[:n_items]  # Only first block
            
    elif field_type == 'vector':
        # Find ALL vector blocks across entire content
        all_vector_blocks = re.findall(r'\(\s*([^)]+?)\s*\)', full_data_block, re.DOTALL)
        result = []
        
        for vec_str in all_vector_blocks:
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', vec_str)
            if len(nums) == 3:
                result.append(tuple(float(x) for x in nums))
        
        if internal_only:
            result = result[:n_items]  # Only first block
    
    else:
        raise ValueError(f"Unsupported field_type: {field_type}")
    
    print(f"Found {len(result)} values (header expected {n_items})")
    return result

def read_boundary_dict(path):
    """
    Read an OpenFOAM constant/polyMesh/boundary file into a Python dict,
    ignoring the FoamFile header and comments.
    """
    with open(path, "r") as f:
        text = f.read()

    # Remove block comments /* ... */ and line comments //
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)

    # Remove FoamFile dictionary completely
    text = re.sub(r"FoamFile\s*\{.*?\}", "", text, flags=re.S)

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    boundary = {}
    i = 0
    while i < len(lines):
        name = lines[i]
        if name in ("(", ")"):
            i += 1
            continue

        # Expect a block starting with '{'
        if i + 1 < len(lines) and lines[i + 1] == "{":
            i += 2
            entry = {}

            while i < len(lines) and lines[i] != "}":
                parts = lines[i].rstrip(";").split()
                if len(parts) == 2:
                    key, val = parts
                    # convert integers where appropriate
                    if val.isdigit():
                        val = int(val)
                    entry[key] = val
                i += 1

            boundary[name] = entry

        i += 1

    return boundary

def load_openfoam_scalar_field(path):
    with open(path, "r") as f:
        text = f.read()

    # Extract content between the first '(' after 'nonuniform' and the matching ')'
    match = re.search(
        r'nonuniform\s+List<scalar>\s*\d+\s*\(\s*(.*?)\s*\)\s*;',
        text,
        re.S
    )

    if match is None:
        raise ValueError("Could not find nonuniform List<scalar> block")

    # Split lines and convert to float
    data = np.array(
        [float(x) for x in match.group(1).split()],
        dtype=float
    )

    return data

def read_vertices(path):
    with open(path, 'r') as f:
        content = f.read()
        
    points_start = content.find('(\n') + 2
    points_text = content[points_start : content.rfind(')')]
        
    points = []
    for line in points_text.strip().split('\n'):
        if line.strip().startswith('('):
            x, y, z = map(float, re.findall(r'[-+]?\d*\.?\d+', line))
            points.append([x, y, z])
    return np.array(points)