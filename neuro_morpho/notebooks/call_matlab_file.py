# Call *.m (Matlab) file

# Requires the following steps before:
# cd /Applications/MATLAB_RXXXX.app/extern/engines/python
# conda activate env
# python3 -m pip install .

import os
import matlab.engine

# Start MATLAB Engine
eng = matlab.engine.start_matlab()

# Resolve the absolute path of the MATLAB script
matlab_script_path = os.path.abspath('../../matlab')

# Add the directory containing the MATLAB file to MATLAB's search path
eng.addpath(matlab_script_path)

# Call the function inside ToolboxAnalysis.m
result = eng.ToolboxAnalysis()  # Pass arguments if needed
#eng.ToolboxAnalysis()

# Print the result
print(result)

# Close the MATLAB session
eng.quit()