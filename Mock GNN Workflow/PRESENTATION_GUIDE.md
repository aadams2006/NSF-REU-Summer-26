# Presentation Guide

This folder contains tools to generate professional presentations of the Mock GNN Workflow project.

## 📊 Available Presentation Formats

### Option 1: PowerPoint (Recommended for Formal Presentations)
**File**: `generate_presentation.py`

**Advantages**:
- Professional appearance
- Easy to edit and customize
- Compatible with all systems
- Print-friendly

**How to Generate**:
```bash
pip install python-pptx
python generate_presentation.py
```

**Output**: `GNN_Workflow_Presentation.pptx`

**Size**: 19 slides covering:
1. Title slide
2. Project overview
3. Why GNNs?
4. Data generation
5. Data characteristics
6. GNN architecture
7. Architecture diagram
8. Training pipeline
9. Results summary
10. Training visualization
11. Model analysis
12. Connection to research
13. Concept transfer
14. Real application example
15. Key learnings
16. Next steps
17. How to use
18. Summary
19. Questions & discussion

---

### Option 2: HTML Slideshow (Best for Web/Portability)
**File**: `generate_html_presentation.py`

**Advantages**:
- Works in any browser
- No software installation needed
- Lightweight and portable
- Animated transitions
- Easy to share

**How to Generate**:
```bash
python generate_html_presentation.py
```

**Output**: `GNN_Workflow_Presentation.html`

**How to View**:
```bash
# Option A: Direct browser open
# Double-click the HTML file in file explorer

# Option B: Local server (better experience)
python -m http.server 8000
# Then open: http://localhost:8000/GNN_Workflow_Presentation.html

# Option C: From terminal
# Windows: start GNN_Workflow_Presentation.html
# macOS: open GNN_Workflow_Presentation.html
# Linux: xdg-open GNN_Workflow_Presentation.html
```

**Keyboard Controls**:
- **Arrow Keys**: Navigate slides
- **ESC**: Overview mode
- **F**: Fullscreen
- **S**: Speaker notes (if available)
- **?**: Help menu

---

## 🚀 Quick Start: Generate Both

```bash
# Generate PowerPoint
python generate_presentation.py

# Generate HTML
python generate_html_presentation.py

# Now you have both formats ready!
```

---

## 📋 Slide Contents

### Slides 1-3: Introduction
- **Slide 1**: Title slide
- **Slide 2**: Project overview & goals
- **Slide 3**: Why GNNs vs traditional methods

### Slides 4-7: Data Generation
- **Slide 4**: Synthetic data creation process
- **Slide 5**: Data statistics and split
- **Slide 6**: GCN architecture overview
- **Slide 7**: Architecture diagram

### Slides 8-11: Training & Results
- **Slide 8**: Training pipeline and optimization
- **Slide 9**: Performance metrics (R², MAE, RMSE)
- **Slide 10**: Training visualization plots
- **Slide 11**: What the model learned

### Slides 12-15: Research Connection
- **Slide 12**: Connection to NSF-REU research
- **Slide 13**: Technique transfer to real project
- **Slide 14**: Real application example
- **Slide 15**: Key learnings demonstrated

### Slides 16-19: Conclusions
- **Slide 16**: Next steps and roadmap
- **Slide 17**: How to execute the project
- **Slide 18**: Summary
- **Slide 19**: Questions & discussion

---

## 🎨 Presentation Design

### Color Scheme
- **Primary**: Dark Blue (#003366)
- **Accent**: Light Blue (#0099CC)
- **Text**: Dark Gray (#333333)
- **Success**: Green (#339966)
- **Background**: White

### Typography
- **Titles**: 40-54pt, Bold
- **Content**: 18-28pt, Regular
- **Code/Details**: 16-20pt, Monospace

### Layout
- **Consistent header bar** with title
- **Bullet points** for readability
- **Two-column sections** for comparison
- **High contrast** for visibility

---

## 📝 Customization Guide

### For PowerPoint (using python-pptx)

Edit `generate_presentation.py`:

**Change colors**:
```python
PRIMARY_COLOR = RGBColor(0, 51, 102)      # Change these RGB values
ACCENT_COLOR = RGBColor(0, 153, 204)
```

**Change slide content**:
```python
add_content_slide(
    "New Title",
    [
        "• Bullet point 1",
        "• Bullet point 2",
        "• Etc."
    ]
)
```

**Add new slides**:
```python
# After last slide
add_content_slide(
    "My New Slide",
    ["Content here"]
)
```

**Change fonts**:
```python
title_para.font.size = Pt(54)  # Adjust font size
title_para.font.bold = True     # Make bold
```

### For HTML (using reveal.js)

Edit `generate_html_presentation.py`:

**Change slide content**:
Modify the HTML between `<!-- Slide X -->` comments

**Change theme**:
Replace `black.min.css` with other reveal.js themes:
- `white.min.css`
- `league.min.css`
- `sky.min.css`
- `beige.min.css`
- `simple.min.css`
- `serif.min.css`
- `blood.min.css`
- `night.min.css`
- `moon.min.css`

**Change animations**:
```javascript
Reveal.initialize({
    transition: 'slide',  // 'slide', 'fade', 'none', 'convex', 'concave', 'zoom'
})
```

---

## 💾 File Sizes & Performance

| Format | File Size | Load Time | Compatibility |
|--------|-----------|-----------|---|
| PowerPoint | ~500 KB | Fast | All systems |
| HTML | ~150 KB | Instant | All browsers |

---

## 📤 Sharing Presentations

### PowerPoint
```bash
# Simply share the .pptx file
# Recipient needs: PowerPoint, Keynote, Google Slides, or LibreOffice
```

### HTML
```bash
# Method 1: Direct file
# Share the .html file (can open anywhere)

# Method 2: Zip with resources (if offline)
# No external dependencies needed

# Method 3: Host online
# Upload to GitHub Pages, Netlify, etc.
```

---

## 🔧 Troubleshooting

### PowerPoint Generation Issues

**Error**: `ModuleNotFoundError: No module named 'pptx'`
```bash
pip install python-pptx
```

**Error**: File encoding issues
```bash
# Ensure UTF-8 encoding
chcp 65001  # Windows
export LC_ALL=en_US.UTF-8  # macOS/Linux
```

### HTML Display Issues

**Issue**: Slides won't advance
- Try different browser (Chrome/Firefox recommended)
- Check browser console (F12) for errors
- Enable JavaScript

**Issue**: Styling looks wrong
- Clear browser cache (Ctrl+Shift+Delete)
- Try fullscreen mode (F)
- Try different browser

**Issue**: Slow on old computer
- Use local server instead of file:// protocol
- Reduce number of slides
- Disable animations

---

## 📊 Presentation Delivery Tips

### Before Presenting
1. Generate both formats as backup
2. Test on the projector/TV
3. Practice smooth transitions
4. Have handouts ready
5. Verify all links work

### During Presentation
1. Use speaker notes (print out)
2. Stand to the side of screen
3. Use pointer to highlight key points
4. Pause for questions
5. Go at your pace (skip slides if needed)

### After Presentation
1. Share the presentation with audience
2. Keep PDF copy for archival
3. Update with feedback
4. Use as portfolio piece

---

## 🎯 Customization Ideas

### Add Personal Touches
- Add your name/institution on title slide
- Include team photos
- Add your research goals
- Custom color scheme

### Enhance Content
- Embed training videos
- Add speaker notes
- Include code snippets
- Link to GitHub repository

### Add Interactive Elements (HTML only)
- Add video links
- Add click-through code examples
- Add pause points for discussion
- Add QR codes

---

## 📚 Presentation Outline Summary

```
INTRODUCTION (3 slides)
├─ Title & Welcome
├─ Project Overview
└─ Motivation (Why GNNs?)

TECHNICAL CONTENT (8 slides)
├─ Data Generation
├─ Architecture Design
├─ Training Process
└─ Results Analysis

RESEARCH CONNECTION (4 slides)
├─ Connection to NSF-REU
├─ Concept Transfer
├─ Real Application
└─ Learning Outcomes

NEXT STEPS (4 slides)
├─ Roadmap
├─ How to Use
├─ Summary
└─ Questions

TOTAL: 19 slides (~25-35 minutes with discussion)
```

---

## ⏱️ Presentation Timing

**Per slide**: 1-2 minutes
**Total slides**: 19
**Estimated time**:
- **Without discussion**: 20-25 minutes
- **With discussion**: 30-40 minutes
- **With detailed questions**: 45-60 minutes

**Recommended segments**:
1. Intro (Slides 1-3): 5 minutes
2. Technical (Slides 4-11): 12 minutes
3. Research (Slides 12-15): 8 minutes
4. Conclusions (Slides 16-19): 5 minutes
5. Questions & Discussion: 10-20 minutes

---

## 🎓 Using as Educational Material

This presentation works well for:
- ✅ Lab meeting presentations
- ✅ Research seminars
- ✅ Conference talks (shorten version)
- ✅ Student teaching
- ✅ Portfolio demonstration
- ✅ Job interview preparation

---

## 📖 Related Documentation

For more details, see:
- [START_HERE.md](START_HERE.md) - Project overview
- [README.md](notes/README.md) - Main documentation
- [GNN_CONCEPTS.md](notes/GNN_CONCEPTS.md) - Theory deep-dive
- [GETTING_STARTED.md](notes/GETTING_STARTED.md) - Setup guide

---

## 🚀 Quick Generation

One-liner to create both presentations:

```bash
python generate_presentation.py && python generate_html_presentation.py && echo "✅ Both presentations ready!"
```

---

## 📞 Support

If presentations don't generate:

1. **Check dependencies**:
   ```bash
   pip list | grep -i pptx
   pip list | grep -i reveal
   ```

2. **Reinstall if needed**:
   ```bash
   pip install --upgrade python-pptx
   ```

3. **Check Python version**:
   ```bash
   python --version  # Should be 3.8+
   ```

4. **Manual HTML creation**: HTML file is pure Python-generated, so if script fails, you can manually create it

---

**Status**: ✅ Ready to Present

Generate presentations with:
```bash
python generate_presentation.py    # PowerPoint
python generate_html_presentation.py  # HTML
```

Then open the generated files and start presenting! 🎉
