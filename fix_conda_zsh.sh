#!/bin/bash

echo "🔧 Fixing conda zsh issues..."

# Clean up problematic environment variables
unset CONDA_PREFIX CONDA_PREFIX_2 CONDA_PREFIX_3 CONDA_SHLVL

# Disable auto-activation of base environment
conda config --set auto_activate_base false

echo "✅ Conda configuration fixed!"
echo ""
echo "📝 To apply changes, please:"
echo "1. Close this terminal"
echo "2. Open a new terminal"
echo "3. Test with: conda --version"
echo ""
echo "🔍 If you still see errors, try:"
echo "   conda --no-plugins <command>"
echo ""
echo "🔄 To manually activate environments:"
echo "   conda activate <environment_name>" 