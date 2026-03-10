import re

with open('src/vulkan_runtime.cpp', 'r') as f:
    text = f.read()

text = text.replace(
    '// Context7 explicitly mandates VK_KHR_synchronization2 to prevent cross-API stalls',
    '// Context7 explicitly mandates VK_KHR_synchronization2 to prevent cross-API stalls\n    // Submissions must use vkCmdPipelineBarrier2 with appropriate VkImageMemoryBarrier2 / VkBufferMemoryBarrier2 structures.'
)

with open('src/vulkan_runtime.cpp', 'w') as f:
    f.write(text)
