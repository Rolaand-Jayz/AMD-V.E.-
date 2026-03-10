import os, glob, re

files = glob.glob('src/**/*.cpp', recursive=True) + glob.glob('include/**/*.hpp', recursive=True)

results = []

for filepath in files:
    with open(filepath, 'r', errors='ignore') as file:
        lines = file.readlines()
    
    deviations = []
    
    content = "".join(lines)
    
    for i, line in enumerate(lines):
        line_idx = i + 1
        line_lower = line.lower()
        
        # 1. Opset > 19 check
        if "opset" in line_lower:
            # Check if it allows higher than 19
            if re.search(r'opset\s*(>|>=|==)\s*(2[0-9]|1[1-9][0-9])', line_lower) or re.search(r'2[0-9]', line_lower):
                if "> 19" not in line and "19" not in line:
                    deviations.append(f"L{line_idx}: Potentially unvalidated opset or allows opset > 19")
            
        # 2. Implicit sync / obsolete barriers
        if re.search(r'vkCmdPipelineBarrier\(', line):
            deviations.append(f"L{line_idx}: Uses deprecated vkCmdPipelineBarrier instead of synchronization2 (vkCmdPipelineBarrier2)")
            
        if re.search(r'vkCmdWaitEvents\(', line):
            deviations.append(f"L{line_idx}: Uses obsolete events sync instead of synchronization2")
            
        # 3. Check for TODOs/stubs related to interop or core logic
        if re.search(r'\b(TODO|FIXME|stub|mock|fake)\b', line, re.IGNORECASE):
            # Only care if it's about our gold standard items
            if re.search(r'vulkan|hip|migraphx|sync|cache|copy', line, re.IGNORECASE):
                snippet = line.strip()[:40]
                deviations.append(f"L{line_idx}: Stub/TODO found in critical path: '{snippet}...'")

    # Full file context checks
    if "migraphx_backend" in filepath.lower():
         if "offload_copy" not in content:
             deviations.append("Missing validation: Does not explicitly encode `offload_copy` status into compilation/cache logic")
         cache_key_components = ["version", "arch", "gpu"]
         if not any(c in content.lower() for c in cache_key_components):
             deviations.append("Missing validation: Model caching lacks strict GPU/Version keying validation")
             
    if "interop" in filepath.lower():
         if "hipImportExternalMemory" not in content and "VK_KHR_external_memory_fd" not in content:
             deviations.append("Missing Validation: Lacks explicit hipImportExternalMemory/VK_KHR_external_memory_fd bindings")
         if "VkSemaphoreSubmitInfo" not in content and "hipImportExternalSemaphore" not in content:
             deviations.append("Missing Validation: Cross-API semaphore synchronization not explicitly enforced")
             
    if "vulkan_runtime" in filepath.lower():
         if "VK_KHR_synchronization2" not in content and "VkImageMemoryBarrier2" not in content:
             deviations.append("Missing Validation: Does not strictly enforce VK_KHR_synchronization2")

    if len(lines) < 5:
        deviations.append("File is effectively empty/stubbed.")

    if deviations:
        # Calculate a rough percentage based on number of deviations + base penalty
        pass_perc = max(0, 100 - (len(deviations) * 15))
        results.append((filepath, f"Fail ({pass_perc}%)", " ; ".join(deviations)))
    else:
        results.append((filepath, "Pass (100%)", "None"))

print("| File | Pass/Fail | Deviating Lines & Context7 Violations |")
print("|---|---|---|")
for r in sorted(results):
    print(f"| {r[0]} | {r[1]} | {r[2]} |")

