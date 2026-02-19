from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
import os

def create_pdf(markdown_file, output_file):
    doc = SimpleDocTemplate(output_file, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_LEFT))
    
    # Custom styles
    title_style = styles["Heading1"]
    h2_style = styles["Heading2"]
    h3_style = styles["Heading3"]
    body_style = styles["Normal"]
    code_style = ParagraphStyle('Code', parent=styles['BodyText'], fontName='Courier', fontSize=9, leading=11)

    story = []

    with open(markdown_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue

        if line.startswith('# '):
            story.append(Paragraph(line.replace('# ', ''), title_style))
            story.append(Spacer(1, 12))
        elif line.startswith('## '):
            story.append(Paragraph(line.replace('## ', ''), h2_style))
            story.append(Spacer(1, 6))
        elif line.startswith('### '):
            story.append(Paragraph(line.replace('### ', ''), h3_style))
            story.append(Spacer(1, 6))
        elif line.startswith('- '):
            story.append(Paragraph(f"• {line.replace('- ', '')}", body_style))
            story.append(Spacer(1, 3))
        elif line.startswith('!['):
            try:
                start = line.find('(') + 1
                end = line.find(')')
                img_path = line[start:end]
                if os.path.exists(img_path):
                    im = Image(img_path, width=400, height=300) # Simple scaling
                    im.hAlign = 'CENTER' # Center alignment
                    story.append(im)
                    story.append(Spacer(1, 12))
            except Exception as e:
                print(f"Image error: {e}")
        elif line.startswith('|'):
             story.append(Paragraph(line, code_style))
        else:
            story.append(Paragraph(line, body_style))
            story.append(Spacer(1, 6))

    doc.build(story)
    print(f"PDF generated: {output_file}")

if __name__ == "__main__":
    create_pdf('report/project_report.md', 'report/RTIE_Project_Report.pdf')
