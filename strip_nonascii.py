
# Strip all non-ASCII bytes from README.md
with open('README.md', 'rb') as f:
    content = f.read()

# Keep only ASCII (0-127)
clean = bytes(b for b in content if b < 128)

with open('README.md', 'wb') as f:
    f.write(clean)

print(f"Cleaned README.md: {len(content)} -> {len(clean)} bytes")
