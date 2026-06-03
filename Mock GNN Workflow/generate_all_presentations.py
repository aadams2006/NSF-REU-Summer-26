"""
Presentation Generator - Create both PowerPoint and HTML presentations in one go.

This script generates:
1. GNN_Workflow_Presentation.pptx (professional PowerPoint)
2. GNN_Workflow_Presentation.html (web-based HTML slideshow)

Run with: python generate_all_presentations.py
"""

import os
import sys
import subprocess

def install_dependencies():
    """Install required dependencies."""
    print("📦 Checking dependencies...\n")
    
    try:
        import pptx
        print("✅ python-pptx is installed")
    except ImportError:
        print("⚠️  python-pptx not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-pptx"])
        print("✅ python-pptx installed successfully\n")


def generate_powerpoint():
    """Generate PowerPoint presentation."""
    print("\n" + "="*60)
    print("🎯 GENERATING POWERPOINT PRESENTATION")
    print("="*60 + "\n")
    
    try:
        print("Generating PowerPoint presentation...")
        
        # Read and execute the PowerPoint generation script with proper context
        with open('generate_presentation.py', 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Create execution context
        exec_globals = {'__name__': '__main__'}
        exec(code, exec_globals)
        
        print("✅ PowerPoint presentation created successfully!\n")
        print("📁 File: GNN_Workflow_Presentation.pptx")
        print("💾 Size: ~500 KB")
        print("📊 Slides: 19")
        
        return True
    except Exception as e:
        print(f"❌ Error generating PowerPoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_html():
    """Generate HTML presentation."""
    print("\n" + "="*60)
    print("🌐 GENERATING HTML PRESENTATION")
    print("="*60 + "\n")
    
    try:
        print("Generating HTML presentation...")
        
        # Read and execute the HTML generation script with proper context
        with open('generate_html_presentation.py', 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Create execution context
        exec_globals = {'__name__': '__main__'}
        exec(code, exec_globals)
        
        print("✅ HTML presentation created successfully!\n")
        print("📁 File: GNN_Workflow_Presentation.html")
        print("💾 Size: ~150 KB")
        print("📊 Slides: 19")
        print("🌐 Open in any browser - no installation needed!")
        
        return True
    except Exception as e:
        print(f"❌ Error generating HTML: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution."""
    print("\n" + "🚀 "*20)
    print("    GNN WORKFLOW PRESENTATION GENERATOR")
    print("🚀 "*20 + "\n")
    
    # Check and install dependencies
    install_dependencies()
    
    # Generate both presentations
    pptx_success = False
    html_success = False
    
    try:
        print("\n" + "="*60)
        print("STEP 1: GENERATING POWERPOINT")
        print("="*60)
        pptx_success = generate_powerpoint()
    except Exception as e:
        print(f"\n⚠️  PowerPoint generation skipped: {e}")
        print("   You can generate it later with: python generate_presentation.py")
    
    try:
        print("\n" + "="*60)
        print("STEP 2: GENERATING HTML")
        print("="*60)
        html_success = generate_html()
    except Exception as e:
        print(f"\n⚠️  HTML generation skipped: {e}")
        print("   You can generate it later with: python generate_html_presentation.py")
    
    # Summary
    print("\n" + "="*60)
    print("✨ GENERATION COMPLETE")
    print("="*60 + "\n")
    
    if pptx_success:
        print("✅ PowerPoint Presentation")
        print("   📁 GNN_Workflow_Presentation.pptx")
        print("   💾 Ready for professional presentations")
        print("   🖥️  Open with: PowerPoint, Keynote, LibreOffice, or Google Slides\n")
    
    if html_success:
        print("✅ HTML Presentation")
        print("   📁 GNN_Workflow_Presentation.html")
        print("   💾 Ready for web sharing")
        print("   🌐 Open with: Any web browser\n")
    
    if not pptx_success and not html_success:
        print("❌ No presentations were generated successfully")
        return 1
    
    # Usage instructions
    print("="*60)
    print("📖 HOW TO USE")
    print("="*60)
    print("\n🎯 For PowerPoint:")
    print("   1. Open: GNN_Workflow_Presentation.pptx")
    print("   2. Edit or customize as needed")
    print("   3. Present with arrow keys")
    print("   4. Share with colleagues")
    
    print("\n🌐 For HTML:")
    print("   Option A (Direct):")
    print("      • Double-click GNN_Workflow_Presentation.html")
    print("   Option B (Local Server - Recommended):")
    print("      • Run: python -m http.server 8000")
    print("      • Open: http://localhost:8000/GNN_Workflow_Presentation.html")
    print("   Controls:")
    print("      • Arrow keys to navigate")
    print("      • ESC for overview")
    print("      • F for fullscreen")
    print("      • ? for help")
    
    print("\n📚 For More Information:")
    print("   • Read: PRESENTATION_GUIDE.md")
    print("   • Details: notes/README.md")
    print("   • Theory: notes/GNN_CONCEPTS.md")
    
    print("\n" + "="*60)
    print("🎉 PRESENTATIONS READY TO GO!")
    print("="*60 + "\n")
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
