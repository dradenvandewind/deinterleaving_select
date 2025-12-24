python -c "
import deinterlace.core
print('Available in deinterlace.core:')
for attr in dir(deinterlace.core):
    if not attr.startswith('_'):
        print(f'  - {attr}')
        
# Try to call hello() if it exists
if hasattr(deinterlace.core, 'hello'):
    print(f\"\\nhello() returns: {deinterlace.core.hello()}\")
"
