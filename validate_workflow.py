#!/usr/bin/env python3
"""
Validates the workflow JSON against ComfyUI's actual node definitions.
Run this after ComfyUI starts to catch any API mismatches before jobs fail.
"""

import json
import urllib.request
import sys
import os

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
WORKFLOW_PATH = "/new_CogVideoX_api.json"


def get_object_info():
    """Fetch node definitions from ComfyUI"""
    url = f"{COMFYUI_URL}/object_info"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read())
    except Exception as e:
        print(f"❌ Failed to fetch object_info: {e}")
        return None


def validate_workflow(workflow: dict, object_info: dict) -> list:
    """Validate workflow nodes against ComfyUI's actual definitions"""
    errors = []
    warnings = []
    
    for node_id, node in workflow.items():
        class_type = node.get("class_type")
        inputs = node.get("inputs", {})
        
        if class_type not in object_info:
            errors.append(f"Node {node_id}: Unknown class_type '{class_type}'")
            continue
        
        node_def = object_info[class_type]
        input_def = node_def.get("input", {})
        required_inputs = input_def.get("required", {})
        optional_inputs = input_def.get("optional", {})
        all_valid_inputs = set(required_inputs.keys()) | set(optional_inputs.keys())
        
        # Check for unknown inputs
        for input_name in inputs.keys():
            if input_name not in all_valid_inputs:
                errors.append(f"Node {node_id} ({class_type}): Unknown input '{input_name}'. Valid inputs: {list(all_valid_inputs)}")
        
        # Check for missing required inputs
        for req_name in required_inputs.keys():
            if req_name not in inputs:
                # Check if it's a connection (list format) that might be set dynamically
                errors.append(f"Node {node_id} ({class_type}): Missing required input '{req_name}'")
        
        # Log node info
        print(f"✓ Node {node_id} ({class_type}): {len(inputs)} inputs validated")
    
    return errors


def main():
    print("=" * 50)
    print("🔍 Validating CogVideoX workflow against ComfyUI...")
    print("=" * 50)
    
    # Load workflow
    try:
        with open(WORKFLOW_PATH) as f:
            workflow = json.load(f)
        print(f"✓ Loaded workflow from {WORKFLOW_PATH}")
        print(f"  Nodes: {list(workflow.keys())}")
    except Exception as e:
        print(f"❌ Failed to load workflow: {e}")
        sys.exit(1)
    
    # Get ComfyUI node definitions
    object_info = get_object_info()
    if not object_info:
        print("⚠️  Could not validate - ComfyUI not responding")
        sys.exit(1)
    
    print(f"✓ Fetched {len(object_info)} node definitions from ComfyUI")
    
    # Print relevant node definitions for debugging
    relevant_nodes = [
        "DownloadAndLoadCogVideoModel",
        "CLIPLoader",
        "CogVideoTextEncode", 
        "CogVideoImageEncode",
        "CogVideoSampler",
        "CogVideoDecode",
        "ImageResizeKJ",
        "LoadImage",
        "VHS_VideoCombine"
    ]
    
    print("\n📋 Relevant node definitions:")
    for node_name in relevant_nodes:
        if node_name in object_info:
            node_def = object_info[node_name]
            input_def = node_def.get("input", {})
            required = list(input_def.get("required", {}).keys())
            optional = list(input_def.get("optional", {}).keys())
            output_types = node_def.get("output", [])
            print(f"\n  {node_name}:")
            print(f"    Required: {required}")
            print(f"    Optional: {optional}")
            print(f"    Outputs: {output_types}")
        else:
            print(f"\n  ❌ {node_name}: NOT FOUND!")
    
    # Validate
    print("\n" + "=" * 50)
    print("🔍 Validating workflow connections...")
    print("=" * 50)
    
    errors = validate_workflow(workflow, object_info)
    
    if errors:
        print("\n❌ VALIDATION FAILED:")
        for error in errors:
            print(f"   • {error}")
        sys.exit(1)
    else:
        print("\n✅ Workflow validation PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    main()

