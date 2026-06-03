import os
import importlib

# Dynamically import all .py files in the utils directory
# Exclude this __init__.py file and any non-Python files
current_dir = os.path.dirname(__file__)

for filename in os.listdir(current_dir):
    if filename.endswith('.py') and filename != '__init__.py':
        module_name = filename[:-3]  # Remove '.py' extension
        importlib.import_module(f'.{module_name}', package='utils')

# Now, all the modules in the utils directory will be available for import.
