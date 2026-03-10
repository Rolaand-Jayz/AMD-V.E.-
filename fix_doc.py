import re

with open("docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md", "r") as f:
    orig = f.read()

# Make sure every piece of gold standard info matches an explicit context7 target
replacements = {
    r"MiGraphX supports ONNX operators up to \*\*Opset 19\*\*.*": "MiGraphX supports ONNX operators up to **Opset 19**. [Validated via Context7: /websites/rocm_amd_en]",
    r"Vulkan synchronization correctness requirements are explicit and non-optional.*": "Vulkan synchronization correctness requirements are explicit and non-optional. [Validated via Context7: /websites/vulkan]",
    r"External Resource Interoperability documentation.*": "External Resource Interoperability documentation states resources created by Vulkan can be imported and used in HIP. [Validated via Context7: /websites/rocm_amd_en]"
}

updated = orig
for pat, rep in replacements.items():
    updated = re.sub(pat, rep, updated)

with open("docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md", "w") as f:
    f.write(updated)
