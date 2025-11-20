"""
Utility functions for formatting and beautification
"""
import re

def format_response(text: str) -> str:
    """
    Beautify and format the response text
    
    Args:
        text: Raw response text
        
    Returns:
        Formatted text with markdown
    """
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Add markdown formatting for lists
    text = re.sub(r'^\d+\.\s+(.+)$', r'1. \1', text, flags=re.MULTILINE)
    
    # Format code blocks (if any)
    text = re.sub(r'```(\w+)?\n(.*?)```', r'```\1\n\2```', text, flags=re.DOTALL)
    
    # Format bold text
    text = re.sub(r'\*\*(.+?)\*\*', r'**\1**', text)
    
    # Format headers
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        # Check if line looks like a header
        if line.strip() and not line.startswith(' ') and len(line) < 100:
            # Check if it's already formatted
            if not line.startswith('#'):
                # Check if it should be a header (short, no punctuation at end)
                if not line.endswith(('.', '!', '?', ':')):
                    formatted_lines.append(f"### {line}")
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    text = '\n'.join(formatted_lines)
    
    # Ensure proper spacing
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def format_timestamp(timestamp: str) -> str:
    """
    Format timestamp for display
    
    Args:
        timestamp: ISO format timestamp
        
    Returns:
        Formatted timestamp string
    """
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp

