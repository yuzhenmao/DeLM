"""Utility functions for SWE-bench."""
from pathlib import Path
from typing import Optional, List


def resolve_path(path_str: str | None, config_path: Path, project_root: Path) -> Path | None:
    """Resolve path relative to config file or project root."""
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    # Try relative to config file first
    relative_to_config = config_path.parent / path
    if relative_to_config.exists():
        return relative_to_config
    # Try relative to project root
    relative_to_root = project_root / path
    if relative_to_root.exists():
        return relative_to_root
    # Return relative to project root even if doesn't exist yet
    return relative_to_root


def truncate_output(output: str, max_chars: int = 10000) -> str:
    """Truncate long output while preserving beginning and end."""
    if len(output) <= max_chars:
        return output
    half = max_chars // 2
    return output[:half] + f"\n\n... [truncated {len(output) - max_chars} chars] ...\n\n" + output[-half:]


def parse_patch(diff_text: str) -> dict:
    """Parse a unified diff into structured patch data."""
    patches = {}
    current_file = None
    current_hunks = []
    
    for line in diff_text.split('\n'):
        if line.startswith('--- '):
            if current_file and current_hunks:
                patches[current_file] = '\n'.join(current_hunks)
            # Extract file path (remove 'a/' prefix if present)
            path = line[4:].strip()
            if path.startswith('a/'):
                path = path[2:]
            current_file = path
            current_hunks = [line]
        elif line.startswith('+++ '):
            current_hunks.append(line)
        elif current_file:
            current_hunks.append(line)
    
    if current_file and current_hunks:
        patches[current_file] = '\n'.join(current_hunks)
    
    return patches


# =============================================================================
# Output Formatting Functions for SWE-Agent ACI
# =============================================================================

def format_file_content(content: str, start_line: int = 1) -> str:
    """
    Format file content with line numbers.
    
    Args:
        content: The file content to format
        start_line: The starting line number
        
    Returns:
        Formatted content with line numbers like "1| def foo():"
        
    Example:
        >>> format_file_content("def foo():\\n    pass", 1)
        '1| def foo():\\n2|     pass'
    """
    if not content:
        return ""
    
    lines = content.split('\n')
    # Remove trailing empty line if present (from trailing newline)
    if lines and lines[-1] == '':
        lines = lines[:-1]
    
    formatted_lines = []
    for i, line in enumerate(lines):
        line_num = start_line + i
        formatted_lines.append(f"{line_num}| {line}")
    
    return '\n'.join(formatted_lines)


def format_command_output(output: str, exit_code: int) -> str:
    """
    Format command output with proper messaging.
    
    - Shows "Command ran successfully. No output." for empty successful commands
    - Returns the output as-is for non-empty output
    
    Args:
        output: The command output
        exit_code: The command exit code
        
    Returns:
        Formatted output string
        
    Example:
        >>> format_command_output("", 0)
        'Command ran successfully. No output.'
        >>> format_command_output("file.txt", 0)
        'file.txt'
    """
    if not output.strip() and exit_code == 0:
        return "Command ran successfully. No output."
    return output


def format_search_results(
    matches: List[tuple], 
    pattern: str, 
    source: str,
    max_results: int = 50
) -> str:
    """
    Format search results in a clean, readable format.
    
    Args:
        matches: List of (line_number, content) tuples
        pattern: The search pattern used
        source: The file/directory searched
        max_results: Maximum results to show
        
    Returns:
        Formatted search results string
    """
    if not matches:
        return f"No matches found for '{pattern}' in {source}"
    
    truncated = len(matches) > max_results
    display_matches = matches[:max_results]
    
    lines = [f"Found {len(matches)} matches for '{pattern}' in {source}:"]
    for line_num, content in display_matches:
        lines.append(f"{line_num}| {content}")
    
    if truncated:
        lines.append(f"... (showing {max_results} of {len(matches)} results)")
    
    return '\n'.join(lines)


def format_file_list(files: List[str], pattern: str, max_files: int = 50) -> str:
    """
    Format a list of files in a clean format.
    
    Args:
        files: List of file paths
        pattern: The search pattern used
        max_files: Maximum files to show
        
    Returns:
        Formatted file list string
    """
    if not files:
        return f"No files found matching '{pattern}'"
    
    truncated = len(files) > max_files
    display_files = files[:max_files]
    
    lines = [f"Found {len(files)} files:"]
    lines.extend(display_files)
    
    if truncated:
        lines.append(f"... (showing {max_files} of {len(files)} files)")
    
    return '\n'.join(lines)


def format_edit_result(
    success: bool,
    file_path: str,
    start_line: int,
    end_line: int,
    syntax_ok: Optional[bool] = None,
    error_message: Optional[str] = None
) -> str:
    """
    Format the result of an edit operation.
    
    Args:
        success: Whether the edit was successful
        file_path: The file that was edited
        start_line: Start line of the edit
        end_line: End line of the edit
        syntax_ok: Whether syntax check passed (None if not applicable)
        error_message: Error message if edit failed
        
    Returns:
        Formatted edit result string
    """
    if not success:
        return f"Edit failed: {error_message or 'Unknown error'}"
    
    result = f"File {file_path} updated (lines {start_line}-{end_line})."
    
    if syntax_ok is not None:
        if syntax_ok:
            result += "\n[Syntax check passed]"
        else:
            result += f"\n[Syntax Error]\n{error_message}"
    
    return result


def format_observation(
    output: str,
    state_info: str,
    step: int,
    max_steps: int
) -> str:
    """
    Format a complete observation for the agent.
    
    Args:
        output: The command output
        state_info: Current state (open file, working directory)
        step: Current step number
        max_steps: Maximum steps allowed
        
    Returns:
        Formatted observation string
    """
    lines = [
        output,
        "",
        state_info,
        f"(Step {step}/{max_steps})",
    ]
    return '\n'.join(lines)
