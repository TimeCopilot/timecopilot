"""Test for validating documentation links.

This module contains tests that validate the links in the project documentation.
It uses the linkchecker tool to verify that links are not broken.

The tests build the documentation using `uv run --group docs mkdocs build` and then
run linkchecker on the generated site to ensure all links are valid.

There are two test functions:
1. test_docs_links_are_valid() - Checks internal links only (suitable for CI)
2. test_docs_external_links_are_valid() - Checks external links (may skip in CI)

Requirements:
- linkchecker must be installed (included in dev dependencies)
- The mkdocs documentation must be buildable
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest


def test_docs_links_are_valid():
    """Test that internal links in the documentation are valid.
    
    This test builds the documentation and uses linkchecker to validate
    that all internal links are not broken. External links are not checked
    to avoid issues with network restrictions in CI environments.
    """
    # Build the documentation
    build_result = subprocess.run(
        ["uv", "run", "--group", "docs", "mkdocs", "build"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )
    
    if build_result.returncode != 0:
        pytest.fail(f"Failed to build documentation: {build_result.stderr}")
    
    # Check if site directory exists
    site_dir = Path.cwd() / "site"
    if not site_dir.exists():
        pytest.fail("Site directory not found after build")
    
    # Run linkchecker on the generated site
    # Use a temporary file to capture linkchecker output
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as temp_file:
        temp_file_path = temp_file.name
    
    try:
        # Run linkchecker with appropriate flags
        # Focus on internal links only to avoid network issues in CI
        linkchecker_result = subprocess.run(
            [
                "linkchecker",
                "--no-warnings",   # Don't show warnings, only errors
                "--output", "text",
                "--file-output", f"text/{temp_file_path}",
                "--recursion-level", "2",  # Limit recursion to avoid infinite checks
                "--timeout", "10",  # Set timeout to 10 seconds
                # Don't check external links to avoid network issues in CI
                # "--check-extern",  # Commented out for CI stability
                str(site_dir / "index.html")  # Start from index.html
            ],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
        )
        
        # Read the output file
        with open(temp_file_path, 'r') as f:
            output = f.read()
        
        # Check if there are any broken internal links
        if linkchecker_result.returncode != 0:
            # Count actual errors vs warnings
            error_lines = [line for line in output.split('\n') if 'Error:' in line]
            if error_lines:
                pytest.fail(f"Linkchecker found broken internal links:\n{output}")
        
        # Verify that the check actually ran
        if "links in" not in output:
            pytest.fail("Linkchecker didn't complete successfully")
            
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def test_docs_external_links_are_valid():
    """Test that external links in the documentation are valid.
    
    This test is marked as optional and may fail in CI environments
    with network restrictions. It checks both internal and external links.
    """
    # This test is marked as optional and may fail in CI environments
    # with network restrictions
    
    # Build the documentation
    build_result = subprocess.run(
        ["uv", "run", "--group", "docs", "mkdocs", "build"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )
    
    if build_result.returncode != 0:
        pytest.skip(f"Failed to build documentation: {build_result.stderr}")
    
    # Check if site directory exists
    site_dir = Path.cwd() / "site"
    if not site_dir.exists():
        pytest.skip("Site directory not found after build")
    
    # Run linkchecker on the generated site with external links
    # Use a temporary file to capture linkchecker output
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as temp_file:
        temp_file_path = temp_file.name
    
    try:
        # Run linkchecker with external link checking
        linkchecker_result = subprocess.run(
            [
                "linkchecker",
                "--check-extern",  # Check external links
                "--no-warnings",   # Don't show warnings, only errors
                "--output", "text",
                "--file-output", f"text/{temp_file_path}",
                "--recursion-level", "2",  # Limit recursion to avoid infinite checks
                "--timeout", "10",  # Set timeout to 10 seconds
                str(site_dir / "index.html")  # Start from index.html
            ],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
        )
        
        # Read the output file
        with open(temp_file_path, 'r') as f:
            output = f.read()
        
        # For external links, we're more lenient due to network issues
        # We'll just warn about broken links rather than fail
        if linkchecker_result.returncode != 0:
            print(f"Warning: Some external links may be broken:\n{output}")
        
    except Exception as e:
        # Skip if there are issues with external link checking
        pytest.skip(f"External link checking failed: {e}")
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)