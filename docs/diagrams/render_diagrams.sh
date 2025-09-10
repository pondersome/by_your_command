#!/bin/bash
# Script to render DOT files to SVG/PNG using Graphviz

# Check if graphviz is installed
if ! command -v dot &> /dev/null; then
    echo "Graphviz not installed. Please install with:"
    echo "  sudo apt-get install graphviz"
    exit 1
fi

# Create output directory
mkdir -p ../images

echo "🎨 Rendering DOT diagrams..."
echo ""

# Render each DOT file to both SVG and PNG
for dot_file in *.dot; do
    if [ -f "$dot_file" ]; then
        base_name=$(basename "$dot_file" .dot)
        
        echo "Processing $base_name..."
        
        # Render to SVG (scalable, preferred for web)
        dot -Tsvg "$dot_file" -o "../images/${base_name}.svg"
        echo "  ✅ Created ../images/${base_name}.svg"
        
        # Render to PNG (fallback for compatibility)
        dot -Tpng "$dot_file" -o "../images/${base_name}.png" -Gdpi=150
        echo "  ✅ Created ../images/${base_name}.png"
        
        echo ""
    fi
done

echo "🎉 All diagrams rendered successfully!"
echo ""
echo "Next steps:"
echo "1. Review the generated SVG files in ../images/"
echo "2. Open in yEd for manual layout optimization if needed"
echo "3. Update README.md to reference the new diagram files"